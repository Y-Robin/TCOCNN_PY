"""Compatibility entry point for the corrected subsensor-0 transfer experiment.

Despite this historical filename, the experiment intentionally uses only the
first subsensor of each of the seven devices.  New code should import
``calibration_transfer_sensor0`` directly.
"""
from calibration_transfer_sensor0 import load_domains, main, scale_domain

__all__ = ['load_domains', 'scale_domain', 'main']


if __name__ == '__main__':
    main()
