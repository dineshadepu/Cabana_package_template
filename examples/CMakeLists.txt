add_executable(Test01ParticlesCreation test_01_particles_creation.cpp)
target_link_libraries(Test01ParticlesCreation PRIVATE cabananewpkg)


add_executable(Test02ComputeNeighbours test_02_compute_neighbours.cpp)
target_link_libraries(Test02ComputeNeighbours PRIVATE cabananewpkg)


add_executable(Test03ComputeVelocityAccelerations test_03_compute_velocity_accelerations.cpp)
target_link_libraries(Test03ComputeVelocityAccelerations PRIVATE cabananewpkg)

add_executable(Grid01 learn_cabana_grid_01_types_example.cpp)
target_link_libraries(Grid01 PRIVATE cabananewpkg)

add_executable(Grid02 learn_cabana_grid_02_global_mesh_example.cpp)
target_link_libraries(Grid02 PRIVATE cabananewpkg)

add_executable(Grid03 learn_cabana_grid_03_partitioner_example.cpp)
target_link_libraries(Grid03 PRIVATE cabananewpkg)

add_executable(Grid04 learn_cabana_grid_04_global_grid_example.cpp)
target_link_libraries(Grid04 PRIVATE cabananewpkg)

add_executable(Grid05 learn_cabana_grid_05_index_space_example.cpp)
target_link_libraries(Grid05 PRIVATE cabananewpkg)

add_executable(TestComparePython01NestedLoop test_compare_python_01_nested_for_loop.cpp)
target_link_libraries(TestComparePython01NestedLoop PRIVATE cabananewpkg)

add_executable(TestComparePython02SerialLoop test_compare_python_02_serial_for_loop.cpp)
target_link_libraries(TestComparePython02SerialLoop PRIVATE cabananewpkg)

add_executable(TestComparePython03Reduce test_compare_python_03_reduce.cpp)
target_link_libraries(TestComparePython03Reduce PRIVATE cabananewpkg)

add_executable(Test04PolicyWithDecreasingLimits test_04_policy_with_decreasing_limits.cpp)
target_link_libraries(Test04PolicyWithDecreasingLimits PRIVATE cabananewpkg)


install(TARGETS Test01ParticlesCreation Test02ComputeNeighbours
  Test03ComputeVelocityAccelerations
  Grid01
  Grid02
  Grid03
  Grid04
  Grid05
  TestComparePython01NestedLoop
  TestComparePython02SerialLoop
  TestComparePython03Reduce
  Test04PolicyWithDecreasingLimits
  DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-w)
