add_executable(Test01ParticlesCreation test_01_particles_creation.cpp)
target_link_libraries(Test01ParticlesCreation PRIVATE cabananewpkg)


add_executable(Test02ComputeNeighbours test_02_compute_neighbours.cpp)
target_link_libraries(Test02ComputeNeighbours PRIVATE cabananewpkg)


install(TARGETS Test01ParticlesCreation Test02ComputeNeighbours
  DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-w)
