# External libraries


## Procedure
We want to know how to integrate a new library into our existing package, such as `Eigen3`.

We need to follow these steps:

- Add the external library in the `CMakeLists.txt` in the root folder
  ```
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  ```
- Compile the entire library, and link it with the external library, i.e., `Eigen3`
  ```
  target_link_libraries(cabananewpkg
  Cabana::Core
  Cabana::Grid
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  )
  ```
  This is done in `CMakeLists.txt` file inside the `src` folder.



## Testing
Inorder to test it, we create a new example file, named `test_08_call_external_libraries_eigen3.cpp`.
Where we will compute the eigenvectors and eigenvalues of a matrix.

Edit the `CMakeLists.txt` to include the example as
```
add_executable(Test08CallExternalLibrariesEigen3 test_08_call_external_libraries_eigen3.cpp)
target_link_libraries(Test08CallExternalLibrariesEigen3 PRIVATE cabananewpkg)

install(TARGETS
  Test08CallExternalLibrariesEigen3
  DESTINATION ${CMAKE_INSTALL_BINDIR})
```


The file contents are as follows:
```cpp
/*
  How to run this examples:
  ./examples/Test08CallExternalLibrariesEigen3
*/
#include <iostream>

#include <CabanaNewPkg.hpp>
#include <Eigen/Dense>



int main( int argc, char* argv[] )
{
  Eigen::Matrix3d I;
  I << 1., 2., 3.,
    33., 11., 232.,
    3., 53., 12.;

  Eigen::EigenSolver<Eigen::Matrix3d> es(I);

  std::cout << "The eigenvalues of I are:\n" << es.eigenvalues() << std::endl;
  std::cout << "The eigenvectors of I are:\n" << es.eigenvectors().col(0).real() << std::endl;

  return 0;
}
```
