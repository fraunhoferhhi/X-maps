<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
<!--   <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
 -->
  <h3 align="center">X-maps: Direct Depth Lookup for Event-based Structured Light Systems</h3>

  <p align="center">
    <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a> -->
    <br />
    <br />
    <a href="https://fraunhoferhhi.github.io/X-maps/"><strong>Project Page</strong></a>
    ·
    <a href="https://fraunhoferhhi.github.io/X-maps/paper-html/x-maps-direct-depth-lookup-for-event-based-structured-light-systems.html" target="_blank"><strong>Paper (HTML)</strong></a>
    ·
    <a href="https://tub-rip.github.io/eventvision2023/papers/2023CVPRW_X-Maps_Direct_Depth_Lookup_for_Event-based_Structured_Light_Systems.pdf" target="_blank"><strong>Paper (PDF)</strong></a>
  </p>

</div>

This project enables you to utilize event cameras to carry out live depth estimations from images projected with a laser projector. We've streamlined the depth estimation process by creating a lookup image with one spatial and one temporal axis (`y` and `t`), forming an X-map. This idea enables speedy depth calculations (taking less than 3 ms per frame), but also maintains the accuracy of depth estimation through disparity search in time maps. The end result is an efficient, reactive tool for designing real-time Spatial Augmented Reality experiences.

The entry point for the live depth estimation is `python/depth_reprojection.py`. The script is using the Metavision SDK to facilitate event data capture from Prophesee cameras. For a straightforward environment setup, an `Ubuntu 20.04` Dockerfile is provided. The depth estimation is implemented in Python with NumPy and Numba.


<!-- ABOUT THE PROJECT -->
<!-- ## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub; however, I didn't find one that really suited my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->



## Getting Started

The project is configured to run from a Docker image in Visual Studio Code (VS Code). It was tested on an Ubuntu 22.04 host.

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:fraunhoferhhi/X-maps.git
   ```
2. Open X-maps folder in VS Code
3. Install the extensions recommended by VS Code (*Docker* and *Dev Containers*)
4. Copy `.devcontainer/metavision.list.template` to `.devcontainer/metavision.list`
5. Edit `.devcontainer/metavision.list` to fill in the URL to the Ubuntu 20.04 Metavision SDK
6. *Reopen in Container* in VS Code

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->

### Download test data

To use data from the [ESL: Event-based Structured Light](https://rpg.ifi.uzh.ch/esl.html) dataset locally:

1. Create a local folder on the host machine to store the data, e.g. `/data/2022_ESL_Event_based_Structured_Light`.
2. Add a `"mounts"` entry in `devcontainer.json`, that mounts the local folder to `/ESL_data`.
3. *Rebuild container* to reopen the project with the mounted folder.
4. Terminal &rarr; Run Task... &rarr; *Download ESL (static) raw and bias files*.

### ESL static depth reprojection

Run the target *X-maps ESL static seq1*. A window should open that performs a live depth estimation of the `book_duck` sequence, projected into the projector's view.

### Live depth reprojection (Spatial Augmented Reality example)

1. Ensure that the camera is working correctly by running `metavision_player` in a Terminal in VS Code.
2. Calibrate your camera-projector setup, and write the parameters into a YAML file, storing the OpenCV matrices. Examples can be found in `data/`.
3. Edit the command line arguments for target *X-maps live depth reprojection* in `.vscode/launch.json`.
4. Display bright content on the projector to allow the start and end of the frame to be identified (trigger finding).
5. Running *X-maps live depth reprojection* creates a window that shows the scene depth from the projector's view.
6. Move the depth reprojection window to the projector to overlay the scene with the measured depth.

To display the depth in full screen on the projector, use the OS window manager to maximize the window. On Ubuntu, a keyboard shortcut can be set under Settings &rarr; Keyboard &rarr; View and Customize Shortcuts &rarr; Windows &rarr; Toggle fullscreen mode.

The parameters you can use when running the `depth_reprojection.py` script can be listed by running `python3 python/depth_reprojection.py --help` in a Terminal in the Docker image in VS Code.

| Parameter  | Explanation |
| ------------- | ------------- |
| `--projector-width`  | Defines the width of the projector in pixels. The default value is `720`.  |
| `--projector-height`  | Defines the height of the projector in pixels. The default value is `1280`.  |
| `--projector-fps`  | Defines the frames per second (fps) of the projector. The default value is `60`.  |
| `--projector-time-map`  | Specifies the path to the calibrated projector time map file (*.npy). If this is left empty, a linear time map will be used. |
| `--z-near`  | Sets the minimum depth in meters (m) for visualization. The default value is `0.1`. |
| `--z-far`  | Sets the maximum depth in meters (m) for visualization. The default value is `1.0`. |
| `--calib`  | Specifies the path to a yaml file with camera and projector intrinsic and extrinsic calibration. This parameter is required. |
| `--bias`  | Specifies the path to the bias file. This is only required for live camera usage. |
| `--input`  | Specifies the path to either a .raw, .dat file for prerecorded sessions. Leave this parameter out for live capture. |
| `--no-frame-dropping` | By default, events are dropped when the processing is too slow. Use this parameter to disable frame dropping, and process all incoming events. |


<!-- ## Technical details -->

<!-- USAGE EXAMPLES -->
<!-- ## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- ROADMAP -->
<!-- ## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

 -->

## License

Distributed under the GPL-3.0 license. See `LICENSE` for more information.


<!-- CONTACT -->
<!-- ## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

 -->

## Acknowledgments

* [ESL: Event-based Structured Light](https://github.com/uzh-rpg/ESL)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
