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
include CMakeFiles/recon_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/recon_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/recon_test.dir/flags.make

CMakeFiles/recon_test.dir/recon_test.cpp.o: CMakeFiles/recon_test.dir/flags.make
CMakeFiles/recon_test.dir/recon_test.cpp.o: ../recon_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/long/Stereo/recon_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/recon_test.dir/recon_test.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/recon_test.dir/recon_test.cpp.o -c /home/long/Stereo/recon_test/recon_test.cpp

CMakeFiles/recon_test.dir/recon_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/recon_test.dir/recon_test.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/long/Stereo/recon_test/recon_test.cpp > CMakeFiles/recon_test.dir/recon_test.cpp.i

CMakeFiles/recon_test.dir/recon_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/recon_test.dir/recon_test.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/long/Stereo/recon_test/recon_test.cpp -o CMakeFiles/recon_test.dir/recon_test.cpp.s

CMakeFiles/recon_test.dir/recon_test.cpp.o.requires:

.PHONY : CMakeFiles/recon_test.dir/recon_test.cpp.o.requires

CMakeFiles/recon_test.dir/recon_test.cpp.o.provides: CMakeFiles/recon_test.dir/recon_test.cpp.o.requires
	$(MAKE) -f CMakeFiles/recon_test.dir/build.make CMakeFiles/recon_test.dir/recon_test.cpp.o.provides.build
.PHONY : CMakeFiles/recon_test.dir/recon_test.cpp.o.provides

CMakeFiles/recon_test.dir/recon_test.cpp.o.provides.build: CMakeFiles/recon_test.dir/recon_test.cpp.o


# Object files for target recon_test
recon_test_OBJECTS = \
"CMakeFiles/recon_test.dir/recon_test.cpp.o"

# External object files for target recon_test
recon_test_EXTERNAL_OBJECTS =

recon_test: CMakeFiles/recon_test.dir/recon_test.cpp.o
recon_test: CMakeFiles/recon_test.dir/build.make
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudabgsegm.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudaobjdetect.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudastereo.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_ml.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_shape.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_stitching.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_superres.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_videostab.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_viz.so.3.2.0
recon_test: /usr/lib/x86_64-linux-gnu/libboost_system.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_thread.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_regex.so
recon_test: /usr/lib/x86_64-linux-gnu/libpthread.so
recon_test: /usr/local/lib/libpcl_common.so
recon_test: /usr/local/lib/libpcl_octree.so
recon_test: /usr/local/lib/libpcl_io.so
recon_test: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
recon_test: /usr/local/lib/libpcl_kdtree.so
recon_test: /usr/local/lib/libpcl_search.so
recon_test: /usr/local/lib/libpcl_visualization.so
recon_test: /usr/local/lib/libpcl_sample_consensus.so
recon_test: /usr/local/lib/libpcl_filters.so
recon_test: /usr/local/lib/libpcl_features.so
recon_test: /usr/local/lib/libpcl_keypoints.so
recon_test: /usr/local/lib/libpcl_surface.so
recon_test: /usr/local/lib/libpcl_registration.so
recon_test: /usr/local/lib/libpcl_ml.so
recon_test: /usr/local/lib/libpcl_segmentation.so
recon_test: /usr/local/lib/libpcl_recognition.so
recon_test: /usr/local/lib/libpcl_people.so
recon_test: /usr/local/lib/libpcl_outofcore.so
recon_test: /usr/local/lib/libpcl_stereo.so
recon_test: /usr/local/lib/libpcl_tracking.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_system.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_thread.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
recon_test: /usr/lib/x86_64-linux-gnu/libboost_regex.so
recon_test: /usr/lib/x86_64-linux-gnu/libpthread.so
recon_test: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
recon_test: /usr/local/lib/libvtkIOMovie-7.1.so.1
recon_test: /usr/local/lib/libvtkoggtheora-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingLOD-7.1.so.1
recon_test: /usr/local/lib/libvtkIOInfovis-7.1.so.1
recon_test: /usr/local/lib/libvtklibxml2-7.1.so.1
recon_test: /usr/local/lib/libvtkGeovisCore-7.1.so.1
recon_test: /usr/local/lib/libvtkproj4-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersSelection-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingVolumeOpenGL2-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersSMP-7.1.so.1
recon_test: /usr/local/lib/libvtkIOPLY-7.1.so.1
recon_test: /usr/local/lib/libvtkIOSQL-7.1.so.1
recon_test: /usr/local/lib/libvtksqlite-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersHyperTree-7.1.so.1
recon_test: /usr/local/lib/libvtkIOImport-7.1.so.1
recon_test: /usr/local/lib/libvtkIOTecplotTable-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingStencil-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersPoints-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersVerdict-7.1.so.1
recon_test: /usr/local/lib/libvtkverdict-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersProgrammable-7.1.so.1
recon_test: /usr/local/lib/libvtkIOParallel-7.1.so.1
recon_test: /usr/local/lib/libvtkIONetCDF-7.1.so.1
recon_test: /usr/local/lib/libvtkjsoncpp-7.1.so.1
recon_test: /usr/local/lib/libvtkIOEnSight-7.1.so.1
recon_test: /usr/local/lib/libvtkIOExport-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingGL2PSOpenGL2-7.1.so.1
recon_test: /usr/local/lib/libvtkViewsContext2D-7.1.so.1
recon_test: /usr/local/lib/libvtkDomainsChemistryOpenGL2-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingContextOpenGL2-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingStatistics-7.1.so.1
recon_test: /usr/local/lib/libvtkIOVideo-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersTexture-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingImage-7.1.so.1
recon_test: /usr/local/lib/libvtkIOLSDyna-7.1.so.1
recon_test: /usr/local/lib/libvtkIOMINC-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersGeneric-7.1.so.1
recon_test: /usr/local/lib/libvtkViewsInfovis-7.1.so.1
recon_test: /usr/local/lib/libvtkIOAMR-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersParallelImaging-7.1.so.1
recon_test: /usr/local/lib/libvtkIOParallelXML-7.1.so.1
recon_test: /usr/local/lib/libvtkInteractionImage-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersFlowPaths-7.1.so.1
recon_test: /usr/local/lib/libvtkIOExodus-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingMorphological-7.1.so.1
recon_test: ../libelas.so
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudafeatures2d.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudacodec.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudaoptflow.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudalegacy.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_calib3d.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudawarping.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_features2d.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_flann.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_highgui.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_objdetect.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_photo.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudaimgproc.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudafilters.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudaarithm.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_video.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_videoio.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_imgcodecs.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_imgproc.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_core.so.3.2.0
recon_test: /home/long/depend/opencv/build/lib/libopencv_cudev.so.3.2.0
recon_test: /usr/local/lib/libvtkgl2ps-7.1.so.1
recon_test: /usr/local/lib/libpcl_common.so
recon_test: /usr/local/lib/libpcl_octree.so
recon_test: /usr/local/lib/libpcl_io.so
recon_test: /usr/local/lib/libpcl_kdtree.so
recon_test: /usr/local/lib/libpcl_search.so
recon_test: /usr/local/lib/libpcl_visualization.so
recon_test: /usr/local/lib/libpcl_sample_consensus.so
recon_test: /usr/local/lib/libpcl_filters.so
recon_test: /usr/local/lib/libpcl_features.so
recon_test: /usr/local/lib/libpcl_keypoints.so
recon_test: /usr/local/lib/libpcl_surface.so
recon_test: /usr/local/lib/libpcl_registration.so
recon_test: /usr/local/lib/libpcl_ml.so
recon_test: /usr/local/lib/libpcl_segmentation.so
recon_test: /usr/local/lib/libpcl_recognition.so
recon_test: /usr/local/lib/libpcl_people.so
recon_test: /usr/local/lib/libpcl_outofcore.so
recon_test: /usr/local/lib/libpcl_stereo.so
recon_test: /usr/local/lib/libpcl_tracking.so
recon_test: ../libelas.so
recon_test: /usr/local/lib/libvtkImagingMath-7.1.so.1
recon_test: /usr/local/lib/libvtkIOGeometry-7.1.so.1
recon_test: /usr/local/lib/libvtkDomainsChemistry-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingOpenGL2-7.1.so.1
recon_test: /usr/lib/x86_64-linux-gnu/libSM.so
recon_test: /usr/lib/x86_64-linux-gnu/libICE.so
recon_test: /usr/lib/x86_64-linux-gnu/libX11.so
recon_test: /usr/lib/x86_64-linux-gnu/libXext.so
recon_test: /usr/lib/x86_64-linux-gnu/libXt.so
recon_test: /usr/local/lib/libvtkglew-7.1.so.1
recon_test: /usr/local/lib/libvtkInfovisLayout-7.1.so.1
recon_test: /usr/local/lib/libvtkViewsCore-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingLabel-7.1.so.1
recon_test: /usr/local/lib/libvtkChartsCore-7.1.so.1
recon_test: /usr/local/lib/libvtkInfovisCore-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingContext2D-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersAMR-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersParallel-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersImaging-7.1.so.1
recon_test: /usr/local/lib/libvtkParallelCore-7.1.so.1
recon_test: /usr/local/lib/libvtkIOLegacy-7.1.so.1
recon_test: /usr/local/lib/libvtkInteractionWidgets-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingVolume-7.1.so.1
recon_test: /usr/local/lib/libvtkIOXML-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersModeling-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingHybrid-7.1.so.1
recon_test: /usr/local/lib/libvtkIOImage-7.1.so.1
recon_test: /usr/local/lib/libvtkDICOMParser-7.1.so.1
recon_test: /usr/local/lib/libvtkmetaio-7.1.so.1
recon_test: /usr/local/lib/libvtkpng-7.1.so.1
recon_test: /usr/local/lib/libvtktiff-7.1.so.1
recon_test: /usr/local/lib/libvtkjpeg-7.1.so.1
recon_test: /usr/lib/x86_64-linux-gnu/libm.so
recon_test: /usr/local/lib/libvtkInteractionStyle-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersExtraction-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersStatistics-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingFourier-7.1.so.1
recon_test: /usr/local/lib/libvtkalglib-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersHybrid-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingAnnotation-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingFreeType-7.1.so.1
recon_test: /usr/local/lib/libvtkRenderingCore-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonColor-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersGeometry-7.1.so.1
recon_test: /usr/local/lib/libvtkfreetype-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingColor-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersSources-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersGeneral-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonComputationalGeometry-7.1.so.1
recon_test: /usr/local/lib/libvtkIOXMLParser-7.1.so.1
recon_test: /usr/local/lib/libvtkIOCore-7.1.so.1
recon_test: /usr/local/lib/libvtkexpat-7.1.so.1
recon_test: /usr/local/lib/libvtkFiltersCore-7.1.so.1
recon_test: /usr/local/lib/libvtkexoIIc-7.1.so.1
recon_test: /usr/local/lib/libvtkNetCDF_cxx-7.1.so.1
recon_test: /usr/local/lib/libvtkNetCDF-7.1.so.1
recon_test: /usr/local/lib/libvtkhdf5_hl-7.1.so.1
recon_test: /usr/local/lib/libvtkhdf5-7.1.so.1
recon_test: /usr/local/lib/libvtkzlib-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingGeneral-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingSources-7.1.so.1
recon_test: /usr/local/lib/libvtkImagingCore-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonExecutionModel-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonDataModel-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonTransforms-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonMisc-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonMath-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonSystem-7.1.so.1
recon_test: /usr/local/lib/libvtkCommonCore-7.1.so.1
recon_test: /usr/local/lib/libvtksys-7.1.so.1
recon_test: CMakeFiles/recon_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/long/Stereo/recon_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable recon_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/recon_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/recon_test.dir/build: recon_test

.PHONY : CMakeFiles/recon_test.dir/build

CMakeFiles/recon_test.dir/requires: CMakeFiles/recon_test.dir/recon_test.cpp.o.requires

.PHONY : CMakeFiles/recon_test.dir/requires

CMakeFiles/recon_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/recon_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/recon_test.dir/clean

CMakeFiles/recon_test.dir/depend:
	cd /home/long/Stereo/recon_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/long/Stereo/recon_test /home/long/Stereo/recon_test /home/long/Stereo/recon_test/build /home/long/Stereo/recon_test/build /home/long/Stereo/recon_test/build/CMakeFiles/recon_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/recon_test.dir/depend
