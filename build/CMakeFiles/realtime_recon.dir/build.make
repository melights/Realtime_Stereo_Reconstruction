# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/long/Stereo/recon_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/long/Stereo/recon_test/build

# Include any dependencies generated for this target.
include CMakeFiles/realtime_recon.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/realtime_recon.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/realtime_recon.dir/flags.make

CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o: CMakeFiles/realtime_recon.dir/flags.make
CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o: ../realtime_recon.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/long/Stereo/recon_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o -c /home/long/Stereo/recon_test/realtime_recon.cpp

CMakeFiles/realtime_recon.dir/realtime_recon.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/realtime_recon.dir/realtime_recon.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/long/Stereo/recon_test/realtime_recon.cpp > CMakeFiles/realtime_recon.dir/realtime_recon.cpp.i

CMakeFiles/realtime_recon.dir/realtime_recon.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/realtime_recon.dir/realtime_recon.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/long/Stereo/recon_test/realtime_recon.cpp -o CMakeFiles/realtime_recon.dir/realtime_recon.cpp.s

CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.requires:

.PHONY : CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.requires

CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.provides: CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.requires
	$(MAKE) -f CMakeFiles/realtime_recon.dir/build.make CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.provides.build
.PHONY : CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.provides

CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.provides.build: CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o


# Object files for target realtime_recon
realtime_recon_OBJECTS = \
"CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o"

# External object files for target realtime_recon
realtime_recon_EXTERNAL_OBJECTS =

realtime_recon: CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o
realtime_recon: CMakeFiles/realtime_recon.dir/build.make
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudabgsegm.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudaobjdetect.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudastereo.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_ml.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_shape.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_stitching.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_superres.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_videostab.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_viz.so.3.2.0
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_system.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_thread.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_regex.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libpthread.so
realtime_recon: /usr/local/lib/libpcl_common.so
realtime_recon: /usr/local/lib/libpcl_octree.so
realtime_recon: /usr/local/lib/libpcl_io.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
realtime_recon: /usr/local/lib/libpcl_kdtree.so
realtime_recon: /usr/local/lib/libpcl_search.so
realtime_recon: /usr/local/lib/libpcl_visualization.so
realtime_recon: /usr/local/lib/libpcl_sample_consensus.so
realtime_recon: /usr/local/lib/libpcl_filters.so
realtime_recon: /usr/local/lib/libpcl_features.so
realtime_recon: /usr/local/lib/libpcl_keypoints.so
realtime_recon: /usr/local/lib/libpcl_surface.so
realtime_recon: /usr/local/lib/libpcl_registration.so
realtime_recon: /usr/local/lib/libpcl_ml.so
realtime_recon: /usr/local/lib/libpcl_segmentation.so
realtime_recon: /usr/local/lib/libpcl_recognition.so
realtime_recon: /usr/local/lib/libpcl_people.so
realtime_recon: /usr/local/lib/libpcl_outofcore.so
realtime_recon: /usr/local/lib/libpcl_stereo.so
realtime_recon: /usr/local/lib/libpcl_tracking.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_system.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_thread.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libboost_regex.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libpthread.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
realtime_recon: /usr/local/lib/libvtkIOMovie-7.1.so.1
realtime_recon: /usr/local/lib/libvtkoggtheora-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingLOD-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOInfovis-7.1.so.1
realtime_recon: /usr/local/lib/libvtklibxml2-7.1.so.1
realtime_recon: /usr/local/lib/libvtkGeovisCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkproj4-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersSelection-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingVolumeOpenGL2-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersSMP-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOPLY-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOSQL-7.1.so.1
realtime_recon: /usr/local/lib/libvtksqlite-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersHyperTree-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOImport-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOTecplotTable-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingStencil-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersPoints-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersVerdict-7.1.so.1
realtime_recon: /usr/local/lib/libvtkverdict-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersProgrammable-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOParallel-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIONetCDF-7.1.so.1
realtime_recon: /usr/local/lib/libvtkjsoncpp-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOEnSight-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOExport-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingGL2PSOpenGL2-7.1.so.1
realtime_recon: /usr/local/lib/libvtkViewsContext2D-7.1.so.1
realtime_recon: /usr/local/lib/libvtkDomainsChemistryOpenGL2-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingContextOpenGL2-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingStatistics-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOVideo-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersTexture-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingImage-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOLSDyna-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOMINC-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersGeneric-7.1.so.1
realtime_recon: /usr/local/lib/libvtkViewsInfovis-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOAMR-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersParallelImaging-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOParallelXML-7.1.so.1
realtime_recon: /usr/local/lib/libvtkInteractionImage-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersFlowPaths-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOExodus-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingMorphological-7.1.so.1
realtime_recon: ../libelas.so
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudafeatures2d.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudacodec.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudaoptflow.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudalegacy.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_calib3d.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudawarping.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_features2d.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_flann.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_highgui.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_objdetect.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_photo.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudaimgproc.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudafilters.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudaarithm.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_video.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_videoio.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_imgcodecs.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_imgproc.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_core.so.3.2.0
realtime_recon: /home/long/depend/opencv/build/lib/libopencv_cudev.so.3.2.0
realtime_recon: /usr/local/lib/libvtkgl2ps-7.1.so.1
realtime_recon: /usr/local/lib/libpcl_common.so
realtime_recon: /usr/local/lib/libpcl_octree.so
realtime_recon: /usr/local/lib/libpcl_io.so
realtime_recon: /usr/local/lib/libpcl_kdtree.so
realtime_recon: /usr/local/lib/libpcl_search.so
realtime_recon: /usr/local/lib/libpcl_visualization.so
realtime_recon: /usr/local/lib/libpcl_sample_consensus.so
realtime_recon: /usr/local/lib/libpcl_filters.so
realtime_recon: /usr/local/lib/libpcl_features.so
realtime_recon: /usr/local/lib/libpcl_keypoints.so
realtime_recon: /usr/local/lib/libpcl_surface.so
realtime_recon: /usr/local/lib/libpcl_registration.so
realtime_recon: /usr/local/lib/libpcl_ml.so
realtime_recon: /usr/local/lib/libpcl_segmentation.so
realtime_recon: /usr/local/lib/libpcl_recognition.so
realtime_recon: /usr/local/lib/libpcl_people.so
realtime_recon: /usr/local/lib/libpcl_outofcore.so
realtime_recon: /usr/local/lib/libpcl_stereo.so
realtime_recon: /usr/local/lib/libpcl_tracking.so
realtime_recon: ../libelas.so
realtime_recon: /usr/local/lib/libvtkImagingMath-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOGeometry-7.1.so.1
realtime_recon: /usr/local/lib/libvtkDomainsChemistry-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingOpenGL2-7.1.so.1
realtime_recon: /usr/lib/x86_64-linux-gnu/libSM.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libICE.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libX11.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libXext.so
realtime_recon: /usr/lib/x86_64-linux-gnu/libXt.so
realtime_recon: /usr/local/lib/libvtkglew-7.1.so.1
realtime_recon: /usr/local/lib/libvtkInfovisLayout-7.1.so.1
realtime_recon: /usr/local/lib/libvtkViewsCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingLabel-7.1.so.1
realtime_recon: /usr/local/lib/libvtkChartsCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkInfovisCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingContext2D-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersAMR-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersParallel-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersImaging-7.1.so.1
realtime_recon: /usr/local/lib/libvtkParallelCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOLegacy-7.1.so.1
realtime_recon: /usr/local/lib/libvtkInteractionWidgets-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingVolume-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOXML-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersModeling-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingHybrid-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOImage-7.1.so.1
realtime_recon: /usr/local/lib/libvtkDICOMParser-7.1.so.1
realtime_recon: /usr/local/lib/libvtkmetaio-7.1.so.1
realtime_recon: /usr/local/lib/libvtkpng-7.1.so.1
realtime_recon: /usr/local/lib/libvtktiff-7.1.so.1
realtime_recon: /usr/local/lib/libvtkjpeg-7.1.so.1
realtime_recon: /usr/lib/x86_64-linux-gnu/libm.so
realtime_recon: /usr/local/lib/libvtkInteractionStyle-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersExtraction-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersStatistics-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingFourier-7.1.so.1
realtime_recon: /usr/local/lib/libvtkalglib-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersHybrid-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingAnnotation-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingFreeType-7.1.so.1
realtime_recon: /usr/local/lib/libvtkRenderingCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonColor-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersGeometry-7.1.so.1
realtime_recon: /usr/local/lib/libvtkfreetype-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingColor-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersSources-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersGeneral-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonComputationalGeometry-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOXMLParser-7.1.so.1
realtime_recon: /usr/local/lib/libvtkIOCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkexpat-7.1.so.1
realtime_recon: /usr/local/lib/libvtkFiltersCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkexoIIc-7.1.so.1
realtime_recon: /usr/local/lib/libvtkNetCDF_cxx-7.1.so.1
realtime_recon: /usr/local/lib/libvtkNetCDF-7.1.so.1
realtime_recon: /usr/local/lib/libvtkhdf5_hl-7.1.so.1
realtime_recon: /usr/local/lib/libvtkhdf5-7.1.so.1
realtime_recon: /usr/local/lib/libvtkzlib-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingGeneral-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingSources-7.1.so.1
realtime_recon: /usr/local/lib/libvtkImagingCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonExecutionModel-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonDataModel-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonTransforms-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonMisc-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonMath-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonSystem-7.1.so.1
realtime_recon: /usr/local/lib/libvtkCommonCore-7.1.so.1
realtime_recon: /usr/local/lib/libvtksys-7.1.so.1
realtime_recon: CMakeFiles/realtime_recon.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/long/Stereo/recon_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable realtime_recon"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/realtime_recon.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/realtime_recon.dir/build: realtime_recon

.PHONY : CMakeFiles/realtime_recon.dir/build

CMakeFiles/realtime_recon.dir/requires: CMakeFiles/realtime_recon.dir/realtime_recon.cpp.o.requires

.PHONY : CMakeFiles/realtime_recon.dir/requires

CMakeFiles/realtime_recon.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/realtime_recon.dir/cmake_clean.cmake
.PHONY : CMakeFiles/realtime_recon.dir/clean

CMakeFiles/realtime_recon.dir/depend:
	cd /home/long/Stereo/recon_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/long/Stereo/recon_test /home/long/Stereo/recon_test /home/long/Stereo/recon_test/build /home/long/Stereo/recon_test/build /home/long/Stereo/recon_test/build/CMakeFiles/realtime_recon.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/realtime_recon.dir/depend
