add_executable(Test01ParticlesCreation test_01_particles_creation.cpp)
target_link_libraries(Test01ParticlesCreation PRIVATE cabananewpkg)


add_executable(Test02ComputeNeighbours test_02_compute_neighbours.cpp)
target_link_libraries(Test02ComputeNeighbours PRIVATE cabananewpkg)


add_executable(Test03ComputeVelocityAccelerations test_03_compute_velocity_accelerations.cpp)
target_link_libraries(Test03ComputeVelocityAccelerations PRIVATE cabananewpkg)


install(TARGETS Test01ParticlesCreation Test02ComputeNeighbours
  Test03ComputeVelocityAccelerations
  DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-w)
