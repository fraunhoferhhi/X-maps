# Produce events either from a raw file, or from a camera with specified biases
# Can't use EventIterator from the Metavision Python samples, as they don't support specifying camera biases

from os import path
import sys
import metavision_hal as mv_hal

from metavision_core.event_io import EventsIterator

from biases import Biases, load_bias_file


class BiasEventsIterator:
    def __init__(self, delta_t, input_filename, bias_file=None):
        # Check validity of input arguments and choose is live camera, .raw file or .dat file
        if not input_filename:
            self.__is_live = True
        elif not (path.exists(input_filename) and path.isfile(input_filename)):
            print("Error: provided input path '{}' does not exist or is not a file.".format(input_filename))
            sys.exit(1)
        else:
            self.__is_live = False

        if self.__is_live:
            # create live camera device interface
            device = mv_hal.DeviceDiscovery.open("")
            # if bias file is provided, load file and set biases in live camera device
            if bias_file:
                biases = Biases(load_bias_file(bias_file))
                for bias in biases.biases:
                    device.get_i_ll_biases().set(bias, biases.biases[bias])
            # if not bias file is provided, the camera uses the default biases
            # to use these biases, a default biases class is created, which initiates with default biases
            else:
                biases = Biases()
            if not device:
                print("No live camera found.")
                sys.exit(1)

            # # Start the camera
            # device.get_i_device_control().start()

            # # Add the device interface to the pipeline to be controlled by that pipeline
            # polling_interval = 0.020  # Interval to poll data from the camera in seconds
            # interface = mvd_core.HalDeviceInterface(device, polling_interval)
            # controller.add_device_interface(interface)
            self.__ev_it = EventsIterator(device, delta_t=delta_t)

        else:
            self.__ev_it = EventsIterator(input_filename, delta_t=delta_t)

    def __iter__(self):
        yield from self.__ev_it

    def get_size(self):
        return self.__ev_it.get_size()
