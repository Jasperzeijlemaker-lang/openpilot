"""
Remote update: lets the "Update Now" switch in sunnylink do what the on-device
"download update" and "install" buttons do, so the owner does not have to be at the car.

Runs only while offroad. It only ever:
  - asks the stock updater to download (the same signal the settings screen sends),
  - sets DoReboot once the stock updater reports a finished update,
  - writes a short status text.
It never restarts the device unless IsOffroad is true at that moment, and it always
clears its own request before asking for a restart, so it cannot restart in a loop.
"""
import subprocess
import time

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

REQUEST_PARAM = "RemoteUpdateNow"
STATUS_PARAM = "RemoteUpdateStatus"
UPDATER_PROC = "openpilot.system.updated.updated"

POLL_TIME = 2.0             # s between checks
NO_RESPONSE_TIME = 150.0    # s to wait for the updater to start working (it sleeps 60 s after boot)
MAX_UPDATE_TIME = 30 * 60.  # s before giving up on a download
SETTLE_TIME = 8.0           # s between "updater is idle again" and reading its result


def signal_updater() -> bool:
  """Ask the stock updater to download. True if the updater process was found."""
  try:
    return subprocess.run(["pkill", "-SIGHUP", "-f", UPDATER_PROC], timeout=10, check=False).returncode == 0
  except Exception:
    cloudlog.exception("remote_update: could not signal updater")
    return False


class RemoteUpdate:
  IDLE = 0
  WAITING = 1

  def __init__(self, params, signal_fn=signal_updater, clock=time.monotonic):
    self.params = params
    self.signal_fn = signal_fn
    self.clock = clock
    self.state = self.IDLE
    self.start_t = 0.
    self.done_t: float | None = None
    self.left_idle = False
    self.last_fetch = None
    self.status = ""

  def _set_status(self, text: str) -> None:
    if text != self.status:
      self.status = text
      self.params.put(STATUS_PARAM, text)

  def _finish(self, text: str) -> None:
    self.state = self.IDLE
    self.done_t = None
    self.left_idle = False
    self.params.put_bool(REQUEST_PARAM, False)
    self._set_status(text)
    cloudlog.warning(f"remote_update: {text}")

  def _restart(self) -> None:
    # the car may have been started since the last check
    if not self.params.get_bool("IsOffroad"):
      self._finish("Car is on. Not restarting.")
      return
    # clear the request first, so a restart can never be requested twice
    self._finish("Restarting to install the update.")
    if self.params.get_bool(REQUEST_PARAM):
      cloudlog.error("remote_update: request did not clear, not restarting")
      return
    self.params.put_bool("DoReboot", True)

  def step(self) -> None:
    try:
      self._step()
    except Exception:
      cloudlog.exception("remote_update: unexpected error")
      try:
        self._finish("Update helper hit an error. Nothing was changed.")
      except Exception:
        self.state = self.IDLE

  def _step(self) -> None:
    requested = self.params.get_bool(REQUEST_PARAM)

    if self.state == self.IDLE:
      if not requested:
        return
      if not self.params.get_bool("IsOffroad"):
        self._finish("Car is on. Updates only run with the car off.")
      elif self.params.get_bool("UpdateAvailable"):
        self._restart()
      elif self.params.get_bool("DisableUpdates"):
        self._finish("'Disable Updates' is on. Turn it off and restart the device.")
      else:
        self.last_fetch = self.params.get("UpdaterLastFetchTime")
        if not self.signal_fn():
          self._finish("Updater is not running. Restart the device and try again.")
          return
        self.state = self.WAITING
        self.start_t = self.clock()
        self.done_t = None
        self.left_idle = False
        self._set_status("Checking for an update...")
      return

    # WAITING
    now = self.clock()
    if not requested:
      self.state = self.IDLE
      self._set_status("Cancelled.")
      return
    if not self.params.get_bool("IsOffroad"):
      self._finish("Car was turned on. Update stopped.")
      return
    if self.params.get_bool("UpdateAvailable"):
      self._restart()
      return

    updater_state = self.params.get("UpdaterState") or ""
    busy = updater_state not in ("", "idle")
    if busy:
      self.left_idle = True
      self.done_t = None
      self._set_status(f"Updater: {updater_state}")
    elif self.left_idle or self.params.get("UpdaterLastFetchTime") != self.last_fetch:
      # the updater writes "idle" a moment before it publishes its result
      if self.done_t is None:
        self.done_t = now
      elif now - self.done_t > SETTLE_TIME:
        failed = self.params.get("UpdateFailedCount") or 0
        if failed:
          self._finish("Download failed. Check Wi-Fi and try again.")
        else:
          self._finish("Already on the newest version.")
        return
    elif now - self.start_t > NO_RESPONSE_TIME:
      self._finish("Updater did not answer. Try again in a minute.")
      return

    if now - self.start_t > MAX_UPDATE_TIME:
      self._finish("Download took too long. Try again later.")


def main() -> None:
  params = Params()
  ru = RemoteUpdate(params)
  while True:
    ru.step()
    time.sleep(POLL_TIME)


if __name__ == "__main__":
  main()
