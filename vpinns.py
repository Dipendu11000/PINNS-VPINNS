#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import numpy as np
import tensorflow as tf

def left_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return np.ones_like(x) * val

def right_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 0.0
    return np.ones_like(x) * val

def top_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 1.0
    return np.ones_like(x) * val

def bottom_boundary(x, y):
    """
    This function will return the boundary value for given component of a boundary
    """
    val = 1.0
    return np.ones_like(x) * val

def rhs(x, y):
    """
    This function will return the value of the rhs at a given point
    """
    # For a Laplace Equation, the rhs function should be zero.
    f_temp = np.ones_like(x) * 0.0
    return f_temp

def exact_solution(x, y):
    """
    This function will return the exact solution at a given point
    """
    # For simplicity, we assume the exact solution is not known.
    # This can be adjusted if an analytical solution is available.
    return np.ones_like(x) * 0.0

def get_boundary_function_dict():
    """
    This function will return a dictionary of boundary functions
    """
    return {1000: bottom_boundary, 1001: right_boundary, 1002: top_boundary, 1003: left_boundary}

def get_bound_cond_dict():
    """
    This function will return a dictionary of boundary conditions
    """
    return {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}

def get_bilinear_params_dict():
    """
    This function will return a dictionary of bilinear parameters
    """
    eps = 1.0
    return {"eps": eps}


# %%




