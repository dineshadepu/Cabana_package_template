add_executable(Test01ParticlesCreation test_01_particles_creation.cpp)
target_link_libraries(Test01ParticlesCreation PRIVATE cabananewpkg)

install(TARGETS Test01ParticlesCreation
  DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-w)
