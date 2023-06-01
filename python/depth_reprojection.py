from metavision_sdk_ui import EventLoop

from bias_events_iterator import BiasEventsIterator, NonBufferedBiasEventsIterator
from depth_reprojection_pipe import DepthReprojectionPipe, RuntimeParams

import click
import sys


@click.command()
@click.option("--projector-width", default=720, help="Projector width in pixels", type=int)
@click.option("--projector-height", default=1280, help="Projector height in pixels", type=int)
@click.option("--projector-fps", default=60, help="Projector fps", type=int)
@click.option(
    "--projector-time-map",
    help="Path to calibrated projector time map file (*.npy). If left empty, a linear time map will be used.",
    type=click.Path(),
)
@click.option("--z-near", default=0.1, help="Minimum depth [m] for visualization", type=float)
@click.option("--z-far", default=1.0, help="Maximum depth [m] for visualization", type=float)
@click.option(
    "--calib",
    help="path to yaml file with camera and projector intrinsic and extrinsic calibration",
    type=click.Path(),
    required=True,
)
@click.option("--bias", help="Path to bias file, only required for live camera", type=click.Path())
@click.option(
    "--input", help="Either a .raw, .dat file for prerecordings. Don't specify for live capture.", type=click.Path()
)
@click.option("--loop-input", help="Loop input file", is_flag=True)
@click.option(
    "--no-frame-dropping", help="Process all events, even when processing lags behind the event stream", is_flag=True
)
def main(bias, input, loop_input, **cli_params):
    print(
        """
Available keyboard options:
- S:     Toggle printing statistics
- Q/Esc: Quit the application"""
    )

    # TODO remove these static values, retrieve from event stream
    params = RuntimeParams(camera_width=640, camera_height=480, **cli_params)

    with DepthReprojectionPipe(params) as pipe:
        mv_iterator = NonBufferedBiasEventsIterator(input_filename=input, delta_t=4000, bias_file=bias)
        # mv_iterator = BiasEventsIterator(input_filename=cli_params["input"], delta_t=8000, bias_file=cli_params["bias"])
        cam_height_reader, cam_width_reader = mv_iterator.get_size()  # Camera Geometry

        assert cam_height_reader == params.camera_height
        assert cam_width_reader == params.camera_width

        for evs in mv_iterator:
            with pipe.stats_printer.measure_time("main loop"):
                # Dispatch system events to the window
                EventLoop.poll_and_dispatch()

                if not len(evs):
                    continue

                pipe.process_events(evs)

                if pipe.should_close():
                    sys.exit(0)


if __name__ == "__main__":
    main()
