//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turb_tube.cpp
//  \brief Problem generator for a flux tube with turbulence generator
//

// C headers

// C++ headers
#include <cmath>
#include <ctime>
#include <sstream>
#include <stdexcept>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

std::int64_t rseed; // seed for turbulence power spectrum

namespace {
// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real R0, b0, lambda;
Real j1zero=3.83170597021;

// functions to compute vector potential to initialize the solution
Real A1(const Real x1, const Real x2, const Real x3);
Real A2(const Real x1, const Real x2, const Real x3);
Real A3(const Real x1, const Real x2, const Real x3);

Real apJ0(const Real x);
Real apJ1(const Real x);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  R0 = pin->GetOrAddReal("problem","R0",1.0);
  b0 = pin->GetOrAddReal("problem","b0",1.0);
  lambda = pin->GetOrAddReal("problem","lambda",j1zero);

  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = pin->GetReal("problem","four_pi_G");
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
  }

  // turb_flag is initialzed in the Mesh constructor to 0 by default;
  // turb_flag = 1 for decaying turbulence
  // turb_flag = 2 for impulsively driven turbulence
  // turb_flag = 3 for continuously driven turbulence
  turb_flag = pin->GetInteger("problem","turb_flag");
  if (turb_flag != 0) {
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    ATHENA_ERROR(msg);
    return;
#endif
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  AthenaArray<Real> a1, a2, a3;
  // nxN != ncellsN, in general. Allocate to extend through ghost zones, regardless # dim
  int nx1 = block_size.nx1 + 2*NGHOST;
  int nx2 = block_size.nx2 + 2*NGHOST;
  int nx3 = block_size.nx3 + 2*NGHOST;
  a1.NewAthenaArray(nx3, nx2, nx1);
  a2.NewAthenaArray(nx3, nx2, nx1);
  a3.NewAthenaArray(nx3, nx2, nx1);

  Real gm1 = peos->GetGamma() - 1.0;
  Real p0 = 1.0;

  int level = loc.level;
  // Initialize components of the vector potential
  if (block_size.nx3 > 1) {
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie+1; i++) {
          if ((pbval->nblevel[1][0][1]>level && j==js)
              || (pbval->nblevel[1][2][1]>level && j==je+1)
              || (pbval->nblevel[0][1][1]>level && k==ks)
              || (pbval->nblevel[2][1][1]>level && k==ke+1)
              || (pbval->nblevel[0][0][1]>level && j==js   && k==ks)
              || (pbval->nblevel[0][2][1]>level && j==je+1 && k==ks)
              || (pbval->nblevel[2][0][1]>level && j==js   && k==ke+1)
              || (pbval->nblevel[2][2][1]>level && j==je+1 && k==ke+1)) {
            Real x1l = pcoord->x1f(i)+0.25*pcoord->dx1f(i);
            Real x1r = pcoord->x1f(i)+0.75*pcoord->dx1f(i);
            a1(k,j,i) = 0.5*(A1(x1l, pcoord->x2f(j), pcoord->x3f(k)) +
                             A1(x1r, pcoord->x2f(j), pcoord->x3f(k)));
          } else {
            a1(k,j,i) = A1(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
          }

          if ((pbval->nblevel[1][1][0]>level && i==is)
              || (pbval->nblevel[1][1][2]>level && i==ie+1)
              || (pbval->nblevel[0][1][1]>level && k==ks)
              || (pbval->nblevel[2][1][1]>level && k==ke+1)
              || (pbval->nblevel[0][1][0]>level && i==is   && k==ks)
              || (pbval->nblevel[0][1][2]>level && i==ie+1 && k==ks)
              || (pbval->nblevel[2][1][0]>level && i==is   && k==ke+1)
              || (pbval->nblevel[2][1][2]>level && i==ie+1 && k==ke+1)) {
            Real x2l = pcoord->x2f(j)+0.25*pcoord->dx2f(j);
            Real x2r = pcoord->x2f(j)+0.75*pcoord->dx2f(j);
            a2(k,j,i) = 0.5*(A2(pcoord->x1f(i), x2l, pcoord->x3f(k)) +
                             A2(pcoord->x1f(i), x2r, pcoord->x3f(k)));
          } else {
            a2(k,j,i) = A2(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
          }

          if ((pbval->nblevel[1][1][0]>level && i==is)
              || (pbval->nblevel[1][1][2]>level && i==ie+1)
              || (pbval->nblevel[1][0][1]>level && j==js)
              || (pbval->nblevel[1][2][1]>level && j==je+1)
              || (pbval->nblevel[1][0][0]>level && i==is   && j==js)
              || (pbval->nblevel[1][0][2]>level && i==ie+1 && j==js)
              || (pbval->nblevel[1][2][0]>level && i==is   && j==je+1)
              || (pbval->nblevel[1][2][2]>level && i==ie+1 && j==je+1)) {
            Real x3l = pcoord->x3f(k)+0.25*pcoord->dx3f(k);
            Real x3r = pcoord->x3f(k)+0.75*pcoord->dx3f(k);
            a3(k,j,i) = 0.5*(A3(pcoord->x1f(i), pcoord->x2f(j), x3l) +
                             A3(pcoord->x1f(i), pcoord->x2f(j), x3r));
          } else {
            a3(k,j,i) = A3(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie+1; i++) {
          if (i != ie+1)
            a1(k,j,i) = A1(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3f(k));
          if (j != je+1)
            a2(k,j,i) = A2(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3f(k));
          if (k != ke+1)
            a3(k,j,i) = A3(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
        }
      }
    }
  }

  // Initialize interface fields
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie+1; i++) {
        pfield->b.x1f(k,j,i) = (a3(k  ,j+1,i) - a3(k,j,i))/pcoord->dx2f(j) -
                               (a2(k+1,j  ,i) - a2(k,j,i))/pcoord->dx3f(k);
      }
    }
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x2f(k,j,i) = (a1(k+1,j,i  ) - a1(k,j,i))/pcoord->dx3f(k) -
                               (a3(k  ,j,i+1) - a3(k,j,i))/pcoord->dx1f(i);
      }
    }
  }

  for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        pfield->b.x3f(k,j,i) = (a2(k,j  ,i+1) - a2(k,j,i))/pcoord->dx1f(i) -
                               (a1(k,j+1,i  ) - a1(k,j,i))/pcoord->dx2f(j);
      }
    }
  }

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = 1.0;

        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = p0/gm1+
              0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                   SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                   SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)))) + (0.5)*
              (SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i))
               + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }
}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
}


namespace {
//----------------------------------------------------------------------------------------
Real apJ0(const Real x) {
  return 1.0/6.0+std::cos(x/2.0)/3.0+std::cos(x*std::sqrt(3.0)/2.0)/3.0+std::cos(x)/6.0;
}

Real apJ1(const Real x) {
  return std::sin(x/2.0)/6.0+std::sin(x*std::sqrt(3.0)/2.0)/2.0/std::sqrt(3.0)+std::sin(x)/6.0;
}

//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//  \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//  Az are functions of x and y alone.

Real A1(const Real x1, const Real x2, const Real x3) {
  Real R = std::sqrt(x1*x1+x2*x2);
  Real sin_ph = x2/R;
  if (R<=R0)
    return -b0*sin_ph*R0/lambda*apJ1(R/R0*lambda);
  else {
    Real Bz0 = apJ0(lambda);
    return -b0*sin_ph*(0.5*R*Bz0-R0*R0*Bz0/2.0/R+R0/lambda*apJ1(lambda)*R0/R);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//  \brief A2: 2-component of vector potential

Real A2(const Real x1, const Real x2, const Real x3) {
  Real R = std::sqrt(x1*x1+x2*x2);
  Real cos_ph = x1/R;
  if (R<=R0)
    return b0*cos_ph*R0/lambda*apJ1(R/R0*lambda);
  else {
    Real Bz0 = apJ0(lambda);
    return b0*cos_ph*(0.5*R*Bz0-R0*R0*Bz0/2.0/R+R0/lambda*apJ1(lambda)*R0/R);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential

Real A3(const Real x1, const Real x2, const Real x3) {
  Real R = std::sqrt(x1*x1+x2*x2);

  if (R<R0)
    return b0*apJ0(R/R0*lambda)*R0/lambda;
  else {
    Real Bph0 = apJ1(lambda);
    return b0*(apJ0(lambda)*R0/lambda-Bph0*R0*std::log(R/R0));
  }
}
} // namespace
