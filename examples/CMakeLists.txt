add_executable(Tst01ParticlesCreation test_01_particles_creation.cpp)
target_link_libraries(Tst01ParticlesCreation PRIVATE cabananewpkg)

install(TARGETS Tst01ParticlesCreation
  DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-w)
