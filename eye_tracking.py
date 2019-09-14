from pathlib import Path
from stytra import Stytra
from stytra.examples.windmill_exp import WindmillProtocol


class TrackingWindmillProtocol(WindmillProtocol):
    name = "windmill"

    # To add tracking to a protocol, we simply need to add a tracking
    # argument to the stytra_config:
    stytra_config = dict(
        tracking=dict(embedded=True, method="eyes"),
        camera=dict(
            type = 'opencv', camera_params = dict(bw = True)
        ),
    )


if __name__ == "__main__":
    s = Stytra(protocol=TrackingWindmillProtocol())