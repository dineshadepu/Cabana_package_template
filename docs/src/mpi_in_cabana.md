# Introduction

We want to learn how to incorporate MPI in Cabana code. This is
helpful to run simulations in a large scale, handle periodic
boundaries.

We follow the following roadmap

1. N-body simulation with MPI in Python [here](#N-body-simulation-with-MPI-in-python)
2. Using Halos in above implementation
3. How Cabana is using MPI, a flowchart and literature survey
4. Apply it in the current code base


# N-body simulation with MPI in python

# How Cabana is using MPI, a flowchart and literature survey

## Halo exchange

1. Use MPI to run massively parallel simualtion, we need Halo, to communicate among different MPI ranks.
2. You can find the corresponding file at `../../examples/test_05_learn_halo_exchange_in_cabana.cpp`.
3. This section is about learning how to use Halo in Cabana.


# CabanaPD mpi flowchart

We take example `elastic_wave.cpp`. The first point of code

```cpp
// line no 47
std::array<double, 3> low_corner = inputs["low_corner"];
std::array<double, 3> high_corner = inputs["high_corner"];
std::array<int, 3> num_cells = inputs["num_cells"];
int m = std::floor( delta /
                    ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
int halo_width = m + 1; // Just to be safe.
```
Here `delta` is the influence radius with which a particle is being
influenced. `halo_width` are the number of additional cells added
around the boundary of a computational domain to account for
interactions that extend beyond the immediate neighboring cells.


Then we send this `halo_width` data into `createParticles` method:
```cpp
// line no 65
auto particles = CabanaPD::createParticles<memory_space, model_type>(
    exec_space(), low_corner, high_corner, num_cells, halo_width );
```
There are a lot of `createParticles` methods in `CabanaPD`
package. But all of these methods essentially create `Particles` class, and initialize it.

In `Particles` initializer, we have

```cpp
// line no 166
// Constructor which initializes particles on regular grid.
template <class ExecSpace>
Particles( const ExecSpace& exec_space, std::array<double, dim> low_corner,
            std::array<double, dim> high_corner,
            const std::array<int, dim> num_cells, const int max_halo_width,
            const std::size_t num_previous = 0,
            const bool create_frozen = false )
    : halo_width( max_halo_width )
    , _plist_x( "positions" )
    , _plist_f( "forces" )
{
    createDomain( low_corner, high_corner, num_cells );
    createParticles( exec_space, num_previous, create_frozen );
}
```
