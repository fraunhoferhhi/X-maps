def load_bias_file(path: str) -> "dict[str,int]":
    """function to load bias values from text file"""
    biases = {}
    with open(path, "r") as file:
        for line in file.readlines():
            line = line.split("%")
            if len(line[0].strip()):
                biases[line[1].strip()] = int(line[0].strip())
    return biases


class Biases:
    """Class to store current camera biases, only works with Gen3.0 cameras"""

    __default_biases = {
        "bias_diff": 300,
        "bias_diff_off": 225,
        "bias_diff_on": 375,
        "bias_fo": 1725,
        "bias_hpf": 1500,
        "bias_pr": 1500,
        "bias_refr": 1500,
    }
    __default_biases_limits = {
        "bias_diff": (300, 300),
        "bias_diff_off": (0, 299),
        "bias_diff_on": (301, 1800),
        "bias_fo": (1650, 1800),
        "bias_hpf": (0, 1800),
        "bias_pr": (1200, 1800),
        "bias_refr": (1300, 1700),
    }

    def __init__(self, biases: "dict[str,int]" = None, biases_limits: "dict[str,tuple[int,int]]" = None):
        """Init"""
        # Check is biases are provided, if not set default biases
        if biases is None:
            self.biases = self.__default_biases
        else:
            self.biases = biases
        # Check is biases are provided, if not set default biases
        if biases_limits is None:
            self.biases_limits = self.__default_biases_limits
        else:
            self.biases_limits = biases_limits

        # Init list with bias names
        self.bias_keys: list[str] = list(self.biases.keys())
        # Init initial trackers for interaction
        self.current_bias_idx: int = 0
        self.current_bias: str = self.bias_keys[self.current_bias_idx]

    def cycle_current_bias(self) -> str:
        """Function to increase currently selected bias for interaction
        and cycle to first if index goes past the last one"""
        self.current_bias_idx += 1
        self.current_bias_idx %= len(self.biases)
        self.current_bias = self.bias_keys[self.current_bias_idx]
        return self.current_bias

    def increase_current(self, step_size=1):
        """Function to increase the currently selected bias by step size"""
        if (self.biases[self.current_bias] + step_size) <= self.biases_limits[self.current_bias][1]:
            self.biases[self.current_bias] += step_size
        else:
            self.biases[self.current_bias] = self.biases_limits[self.current_bias][1]
        return self.biases[self.current_bias]

    def decrease_current(self, step_size=1):
        """Function to decrease the currently selected bias by step size"""
        if (self.biases[self.current_bias] - step_size) >= self.biases_limits[self.current_bias][0]:
            self.biases[self.current_bias] -= step_size
        else:
            self.biases[self.current_bias] = self.biases_limits[self.current_bias][0]
        return self.biases[self.current_bias]
