/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaNewPkg.hpp>

namespace Test
{
//---------------------------------------------------------------------------//
void testParticlesCreation()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = typename exec_space::memory_space;

    auto particles = std::make_shared<
      CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1);
    particles->resize(20);
    auto x_p = particles->slicePosition();
    EXPECT_DOUBLE_EQ(20, x_p.size());
}


void testParticlesOutputFolderCreationFromInitializer()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = typename exec_space::memory_space;

    auto particles = std::make_shared<
      CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1,
                                                  "test_particles_output_folder_creation_output");

    EXPECT_EQ(fs::exists("test_particles_output_folder_creation_output"), true);
    EXPECT_EQ(fs::is_directory("test_particles_output_folder_creation_output"), true);

    if (fs::exists("test_particles_output_folder_creation_output")) {
      // Delete the folder and all its contents
      fs::remove_all("test_particles_output_folder_creation_output");
    }
}


void testParticlesOutputFolderCreationFromSetOutputFolder()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = typename exec_space::memory_space;

    auto particles = std::make_shared<
      CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1);
    particles->set_output_folder("test_particles_output_folder_creation_2_output");

    EXPECT_EQ(fs::exists("test_particles_output_folder_creation_2_output"), true);
    EXPECT_EQ(fs::is_directory("test_particles_output_folder_creation_2_output"), true);

    if (fs::exists("test_particles_output_folder_creation_2_output")) {
      // Delete the folder and all its contents
      fs::remove_all("test_particles_output_folder_creation_2_output");
    }
}


void testParticlesHdfFile()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = typename exec_space::memory_space;

    auto particles = std::make_shared<
      CabanaNewPkg::Particles<memory_space, DIM>>(exec_space(), 1);
    particles->set_output_folder("test_particles_output_folder_creation_3_output");
    particles->output(0, 0);

    EXPECT_EQ(fs::exists("test_particles_output_folder_creation_3_output/particles_0.h5"), true);
    // EXPECT_EQ(fs::is_directory("test_particles_output_folder_creation_2_output"), true);


    if (fs::exists("test_particles_output_folder_creation_3_output")) {
      // Delete the folder and all its contents
      fs::remove_all("test_particles_output_folder_creation_3_output");
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_particles_creation )
{
    testParticlesCreation();
}

TEST( TEST_CATEGORY, test_particles_output_folder_creation_methods )
{
    testParticlesOutputFolderCreationFromInitializer();
    testParticlesOutputFolderCreationFromSetOutputFolder();
}


TEST( TEST_CATEGORY, test_particles_output_hdf_file )
{
  testParticlesHdfFile();
}

//---------------------------------------------------------------------------//

} // end namespace Test
