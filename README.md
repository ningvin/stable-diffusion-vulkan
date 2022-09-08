*Looking for the original README? You can find it [here](ORIGINAL_README.md).*

My fork of Stable Diffusion that is supposed to run on Vulkan. Spoiler: I could not get it to work.

# 1 - Introduction

## 1.1 - Why?

Seemed like a fun idea. Also, these are my specs:
- AMD Ryzen 5 2600
- AMD Radeon RX 580
- 16 GiB RAM
- Pop!_OS / Ubuntu 22.04

So CUDA is out, ROCm is out ([unless maybe with arcane knowledge](https://www.reddit.com/r/StableDiffusion/comments/ww436j/howto_stable_diffusion_on_an_amd_gpu/)) and running stable diffusion on the CPU takes 12 minutes for a single image...

But then I read somewhere that PyTorch has a [Vulkan backend](https://pytorch.org/tutorials/prototype/vulkan_workflow.html), so that should work and be fast, right?

## 1.2 - Disclaimer

I do not know what I am doing for the most part. You have been warned.

All this was done on Pop!_OS / Ubuntu with above specs. If you are running a different setup, YMMV.

# 2 - General Setup

## 2.1 - Prerequisites

- Python (I used 3.10.4)
- C++ compiler like GCC or Clang (I used GCC 11.2.0)
- Vulkan SDK (see instructions [here](https://vulkan.lunarg.com/sdk/home#linux))

Probably a bunch of other stuff I forgot.

## 2.2 - Preparing a Workspace

We will be dealing with lots of python dependencies. The official Stable Diffusion readme suggests using `conda`, but I ended up using `pip` and virtual environments

This is the folder structure I came up with:

```
root // e.g. ~/Workspace/stable-diffusion
  |
  +-- .venv // for collecting all dependencies required by stable diffusion
  |
  +-- pytorch // the official PyTorch repo, setup in a later step
  |
  +-- stable-diffusion-vulkan // this repository, setup in a later step
```

Create a virtual environment like this:

```bash
me:root$ python3 -m venv .venv
```

Activate it:

```bash
me:root$ source .venv/bin/activate
```

I will try to remind you whenever this virtual environment needs to be active throughout this odyssey.

# 3 - Setting up the Stable Diffusion

## 3.1 - Setting up the Repository

Clone this repository:

```bash
me:root$ git clone https://github.com/ningvin/stable-diffusion-vulkan.git
```

Follow the steps provided by the [original README](ORIGINAL_README.md#reference-sampling-script) to download the weights.

## 3.2 - Installing the Dependencies

Make sure the virtual environment created in [2.2](#22---preparing-a-workspace) is active. The dependencies required for stable diffusion are listed [here](environment.yaml).

I ended up skipping the following dependencies:
- `pytorch` (we will be building that ourselves in a later step)
- `cudatoolkit` (we are not using CUDA)

This left me with the requirements in the [requirements.txt](requirements.txt) file. They can be installed like this:

```bash
(.venv) me:root$ pip install -r requirements.txt
```

Additionally, we need to install the following with the `-e` switch:

```bash
(.venv) me:root$ pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
(.venv) me:root$ pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
(.venv) me:root$ pip install -e .
```

**Note:** one of the steps above probably also installed a version of PyTorch as an indirect dependency. As we will build our own later, we need to uninstall this one:

```bash
(.venv) me:root$ pip uninstall pytorch
```

# 4 - Setting up PyTorch

It turns out that while PyTorch has a Vulkan backend, they do not ship a build of it. So we have to build it ourselves.

## 4.1 - Cloning the Repository

Inside the root folder, execute the following:

```bash
me:root$ git clone --recursive https://github.com/pytorch/pytorch
```

**Note:** the repo is quite large and has a ton of submodules. This might take some time...

## 4.2 - Installing the Dependencies

With the virtual environment created in [2.2](#22---preparing-a-workspace) active:

```bash
(.venv) me:root$ pip install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses mkl mkl-include
```

These dependencies are taken from [here](https://github.com/pytorch/pytorch#install-dependencies). Some of them might be already satisfied by the ones installed for stable diffusion.

## 4.3 - Building PyTorch with the Vulkan Backend enabled

Make sure the virtual environment created in [2.2](#22---preparing-a-workspace) is active.

The [Vulkan Workflow documentation]((https://pytorch.org/tutorials/prototype/vulkan_workflow.html)) suggests the following:

```bash
# BAD:
USE_VULKAN=1 USE_VULKAN_SHADERC_RUNTIME=1 USE_VULKAN_WRAPPER=0
```

I initially tried that and ran into some build errors (unescaped `"` in a C-string inside a generated shader source). After fixing the issue and completing the build successfully, I ran into issues further down the road: for some reason, PyTorch seemed to pass an empty shader to the driver at some point, which the latter rejected.

So I ended up ditching the `USE_VULKAN_SHADERC_RUNTIME` variable. Here is my complete build command which should be executed from the `pytorch` directory:

```bash
# BETTER:
(.venv) me:root/pytorch$ USE_VULKAN=1 USE_VULKAN_WRAPPER=0 USE_CUDA=0 USE_ROCM=0 python setup.py develop
```

**Note:** I am using the `develop` command instead of the `install` command the documentation suggests as the latter caused some errors.

**Warning:** it took quite a while on my system until the build was finished. Also, the whole build process is quite taxing on RAM. If you encounter that it requires too much RAM at once, you may decrease the build parallelization by setting the `MAX_JOBS` environment variable to a lowish value (`MAX_JOBS=1` for a sequential build).

You can check if everything worked by spinning up a python interactive shell (again, with the virtual environment active):

```
(.venv) me:root/pytorch$ python
Python 3.10.4 (main, Jun 29 2022, 12:14:53) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.is_vulkan_available()
True
>>> exit()
```

If `is_vulkan_available` returns anything other than `True`, it did in fact not work.

# 5 - Trying to run Stable Diffusion on Vulkan

Congrats, you made it this far! PyTorch has been built, Stable Diffusion has been set up and forced to be friends with Vulkan, what could go wrong?

## 5.1 - Finding the correct Vulkan Device

Turns out a lot. Chances are that even if you have only one graphics card, there are two Vulkan devices available on your system:
- your actual dedicated graphics card
- [llvmpipe](https://docs.mesa3d.org/drivers/llvmpipe.html), which emulates everything on the CPU

And guess which device was selected by default on my setup...

So if you actually want to target your graphics card, better specify it explicitly using the `DRI_PRIME` environment variable. To figure out your device's device id and vendor id, use the `vulkaninfo` utility (should have shipped with your Vulkan SDK install). It will produce a wall of text, you are looking for the *"Device Properties and Extensions"* section:

```
me:root$ vulkaninfo
...
Device Properties and Extensions:
=================================
GPU0:
VkPhysicalDeviceProperties:
---------------------------
    ...
    vendorID          = 0x1002  // might be different for your setup
    deviceID          = 0x67df  // might be different for your setup
    deviceType        = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU  // make sure you are looking at your actual graphics card
    ...
...
```

So in my case, I use the following:

```bash
DRI_PRIME="1002:67df"  # also possible: "0x1002:0x67df"
```

## 5.2 - Running Stable Diffusion

Make sure the virtual environment created in [2.2](#22---preparing-a-workspace) is active. From within the folder you cloned Stable Diffusion into, execute the following:

```bash
(.venv) me:root/stable-diffusion-vulkan$ DRI_PRIME="1002:67df" python ./scripts/txt2img.py --prompt "my friend the tree is hugging me" --n_samples 1 --n_iter 1 --plms
```

**Note:** obviously substitute the value for `DRI_PRIME` with whatever you found out in [5.1](#51---finding-the-correct-vulkan-device), as well as the prompt with whatever your imagination can come up with :^)

**Warning:** this uses quite the amount of RAM while loading the weights and processing the model. You might want to have an eye on your RAM usage and kill other processes that might use lots of RAM, like browsers, VS Code, etc. Unless you have more than 16 GiB of RAM, then you are probably fine...

If this actually works for you and produces a picture: congrats! If not: welcome to the club!

I ran into a lot of problems along the way:
- segmentation fault (remember the bad build flag in [4.3](#43---building-pytorch-with-the-vulkan-backend-enabled)? yeah that one...)
- killed (had to use `journalctl` to figure out it was due to using too much RAM; took a while to figure out that this was due to using llvmpipe, hence [5.1](#51---finding-the-correct-vulkan-device))
- error return code / exception

I have tried a bunch of things, but I am stuck at this last category of errors. The next section is a summary of those things.

# 6 - Troubleshooting

This assumes you have made it to [5.2](#52---running-stable-diffusion) without any hickups, but are unable to run Stable Diffusion.

## 6.1 - Dealing with Core Dumps

In case you are running into a segmentation fault, your system can generate a core dump which you can analyze. This may be suppressed by your current `ulimit` settings:

```bash
me:root$ ulimit -c
```

If the above call returns `0`, you can lift that restriction by executing the following:

```bash
me:root$ ulimit -c unlimited
```

Where the resulting core file is written to is actually a science of its own. Google is your friend here. For recent Ubuntu based systems `/var/lib/apport/coredump` seems to be the location.

You can also use some utility programs to manually generate a core dump of a running process, e.g. `gcore` (**Note:** might require `sudo`).

Once you have obtained a core dump, you can analyze it with GDB:

```bash
me:root$ gdb python /path/to/my/core/file
```

In case you see a lot of blanks and question marks when querying the callstack, e.g. using the `where` command, you may want to download additional debug symbols.

## 6.2 - Obtaining additional Debug Symbols

On Ubuntu based systems, follow [this guide](https://wiki.ubuntu.com/Debug%20Symbol%20Packages) to setup the debug symbol repository.

I for my part installed `mesa-vulkan-drivers-dbgsym`.

## 6.3 - Debugging PyTorch using GDB

Kind of a bad idea. First, you need to (re)compile PyTorch with debug symbols. Simply add `DEBUG=1` to the build command in [4.3](#43---building-pytorch-with-the-vulkan-backend-enabled) (you might want to clean the `build` folder before that though).

I was then unable to launch the script from within GDB:
```bash
(.venv) me:root/stable-diffusion-vulkan$ DRI_PRIME="1002:67df" gdb python
...
(gdb) run ./scripts/txt2img.py --prompt "my friend the tree is hugging me" --n_samples 1 --n_iter 1 --plms
```

After executing the `run` command it would create some threads and then pretty much nothing happened after that.

I had more success attaching to an already running process. Simply run Stable Diffusion as described in [5.2](#52---running-stable-diffusion), wait until the model is loaded (you may add a simple `input()` prompt into `txt2img.py` right before the data is sent to the Vulkan device) and then execute the following in a different terminal:

```bash
me:root/stable-diffusion-vulkan$ sudo gdb attach {pid}
```

where `{pid}` is the process id of the `python` process.

It was usable, but way slower, used arguably more RAM (which apparently is a scarce resource on my system) and had no real benefit compared to print debugging.

## 6.4 - Debugging PyTorch using `printf`

Once I overcame the segmentation faults and shady default Vulkan devices, I was left with an exception originating from the [Vulkan Memory Allocator](https://gpuopen.com/vulkan-memory-allocator/), [vk_mem_alloc.h](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/blob/a6bfc237255a6bac1513f7c1ebde6d8aed6b5191/include/vk_mem_alloc.h) (located in the `third_party` folder of the PyTorch repo) to be specific.

I quickly noticed the `VMA_DEBUG_LOG` macro sprinkled throughout the file, which is defined empty by default. Changing its definition to the following and rebuilding PyTorch yielded some more diagnostics output:

```cpp
#ifndef VMA_DEBUG_LOG
//    #define VMA_DEBUG_LOG(format, ...)
   #define VMA_DEBUG_LOG(format, ...) do { \
       fprintf(stderr, "[VMA_DBG]: " format, ##__VA_ARGS__); \
       fprintf(stderr, "\n"); \
   } while(false)
#endif
```

**Note:** I am using the GCC (and maybe also Clang) specific `##` trick to deal with the dangling `,` after `format` for cases where no format arguments are given. In case this does not work on your compiler you might have to get creative and/or consult your local macro guru.

You can also add some more calls to `VMA_DEBUG_LOG`, e.g. to trace return values of certain Vulkan functions.

All in all, this has been the most helpful debugging tool.

## 6.5 - Debugging Mesa using `printf`

After analyzing the diagnostics output obtained in [6.4](#64---debugging-pytorch-using-printf), I tried to get a better look into what the Mesa driver was doing, specifically `libvulkan_radeon.so`.

Debugging using GDB seemed out of the question, so I tried to follow a similar approach as in [6.4](#64---debugging-pytorch-using-printf). For this I loosely followed the instructions [here](https://docs.mesa3d.org/install.html#running-against-a-local-build).

While I was able to build the Vulkan driver and add diagnostic traces (`printf`) to it, I did not have much success using it to track down problems. The drivers built by me (both debug and optimized) always seemed to use *way* more RAM than the ones that shipped with Pop!_OS / Ubuntu, so much in fact that the system ran out of RAM before it even got to the critical part...

### 6.5.1 - Cloning the Repository

My system is running Mesa 22.0.5 by default. I tried using both the latest version of Mesa, as well as the 22.0.5 tag. In case you want to use the latest version:

```bash
me:root$ git clone --depth 1 https://gitlab.freedesktop.org/mesa/mesa.git
```

### 6.5.2 - Installing the Dependencies

Mesa requires a bunch of dependencies. First and foremost, the [Meson build system](https://mesonbuild.com/). It can be installed via `pip`. I ended up creating a separate virtual environment for the Mesa repository:

```bash
me:root/mesa$ python3 -m venv .mesa
me:root/mesa$ source .mesa/bin/activate
(.mesa) me:root/mesa$ pip install meson
```

In order to build Mesa, I also needed to install the following dependencies with `apt`:

```
libxcb-dri2-dev
libxcb-dri3-dev
libxcb-present-dev
libxshmfence-dev
bison
flex
```

This list may be incomplete. If something is missing on your system, Meson will let you know.

### 6.5.3 - Building the Vulkan Radeon Driver

Make sure the virtual environment created in [6.5.2](#652---installing-the-dependencies) is active. Inside the Mesa repository, create a folder called `build` and change into it.

I initially used the following command to generate the build files:

```bash
(.mesa) me:root/mesa/build$ meson .. -Dprefix="root/mesa/install" -Ddri-drivers= -Dgallium-drivers= -Dvulkan-drivers=amd
```

**Note:** obviously substitute the `prefix` variable with a suitable path. Setting this variable ensures that you do not replace the driver that is currently used by your system.

This by default builds a debug version of the driver. In case you want an optimized build, additionaly specify the `buildtype`:

```bash
(.mesa) me:root/mesa/build$ meson .. -Dprefix="root/mesa/install" -Ddri-drivers= -Dgallium-drivers= -Dvulkan-drivers=amd -Dbuildtype=release
```

Also, this targets `ninja` by default. If you want to use something different to build your code, e.g. `make`, you can probably specify that as well.

After the build files have been generated, simply run:

```bash
(.mesa) me:root/mesa/build$ ninja install
```

### 6.5.4 - Overriding the Vulkan Driver at Runtime

The above should have generated the following two files, among other things:
- `libvulkan_radeon.so`
- `radeon_icd.x86_64.json` (should contain a refernce to `libvulkan_radeon.so`)

We can specify the latter when running Stable Diffusion by using the `VK_ICD_FILENAMES` environment variable. Such a call to Stable Diffusion might look like this:

```bash
(.venv) me:root/stable-diffusion-vulkan$ DRI_PRIME="1002:67df" VK_ICD_FILENAMES="root/mesa/install/share/vulkan/icd.d/radeon_icd.x86_64.json" python ./scripts/txt2img.py --prompt "my friend the tree is hugging me" --n_samples 1 --n_iter 1 --plms
```

**Note:** ensure that the correct virtual environment is active.

And voilà: Stable Diffusion should use the Vulkan driver you have built in [6.5.3](#653---building-the-vulkan-radeon-driver). Just add some `printf` calls, rebuild (make sure you build the `install` target) and you have a driver with fancy diagnostics output.

## 7 - My current Status

I get to a point where the 733rd call to `vmaCreateImage` reproducibly throws an exception, due to an error code returned by a call to `VmaAllocator_T::AllocateMemory`.

I was suspecting memory shortage on my GPU at first, but thanks to `printf` I found out that the previous call to `VmaAllocator_T::GetImageMemoryRequirements` (which internally calls `vkGetImageMemoryRequirements`) claims it requires 0 memory, which upsets the allocator.

Why this image is supposed to occupy no memory is beyond me. It is created with the same flags as the 732 images before it and has a size of `3 x 3 x 65535` if I recall correctly. It is not even the largest image created.

The implementation of `vkGetImageMemoryRequirements` inside Mesa seems to simply return a value already calculated during `vkCreateImage`. But from here on I was unable to follow things further.

## 8 - Conclusion

Maybe this writeup is helpful to someone. Maybe I made a small mistake along the way and there is a simple solution to all of this. Maybe my setup is borked in some way. Who knows. I surely do not.

On the positive side of things, I learned a lot. Did you know for example that the Mesa project indents their C/C++ code using [3 spaces](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/.editorconfig#L13)?

Oh, and special thanks to Louis Cole who has kept me alive through all of this!
