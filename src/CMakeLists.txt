set(HEADERS
  CabanaNewPkg_Particles.hpp
  CabanaNewPkg.hpp
  )

set(SOURCES
  CabanaNewPkg_Particles.cpp
  )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(cabanaNewPkg ${SOURCES})

target_link_libraries(cabanaNewPkg
  Cabana::Core
  Cabana::Grid
  nlohmann_json::nlohmann_json
  )

target_include_directories(cabanaNewPkg
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

install(TARGETS cabanaNewPkg LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
