# CMake entry point
cmake_minimum_required (VERSION 2.8)
project (rectify)

set(OpenCV_DIR "/home/long/depend/opencv/build")

find_package(OpenCV REQUIRED)
find_package( PCL REQUIRED )

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(recon_test
	recon_test.cpp
)
target_link_libraries(recon_test
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${PROJECT_SOURCE_DIR}/libelas.so
)

add_executable(realtime_stereo
	realtime_stereo.cpp
)
target_link_libraries(realtime_stereo
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${PROJECT_SOURCE_DIR}/libelas.so
)

add_executable(realtime_recon
	realtime_recon.cpp
)
target_link_libraries(realtime_recon
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${PROJECT_SOURCE_DIR}/libelas.so
)

add_executable(realtime_surface
	realtime_surface.cpp
)
target_link_libraries(realtime_surface
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${PROJECT_SOURCE_DIR}/libelas.so
)
