cmake_minimum_required( VERSION 3.16 )

project( magiceye )
set( CMAKE_CXX_STANDARD 17 )

find_package( OpenCV REQUIRED )

set( json_INCLUDE_DIRS
     json/
     )

set( SRC_FILES
     src/CardDatabase.cpp
     src/CardFinder.cpp
     src/DeltaEGrid.cpp
     src/HistoGrid.cpp
     src/jsoncpp.cpp
     src/MagicCard.cpp
     src/MagicEyeGUI.cpp
     src/main.cpp
     src/TableCard.cpp
     )

add_executable( ${PROJECT_NAME} ${SRC_FILES} )
target_include_directories( ${PROJECT_NAME} PRIVATE include
                                            PRIVATE ${json_INCLUDE_DIRS}
                                            PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( ${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )
target_compile_options( ${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror )
