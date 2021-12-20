# The MPC tracking for quadrotor

The controller for quadrotor tracking are include 3 subcontrollers. They are Altititude controller , Position controller and Attitude controller. The detail of them are shown in the Figure below.

![MPC quad](images/quad_controller.png)

The MPC solver is using [CasADi](https://web.casadi.org/).
## The tracking of each controller
Implement 3 MPC controller without relation

### The altitude tracking
![altitude](images/altitude_tracking.png)

### The position tracking
![position](images/position_tracking.png)

### The attitude tracking
![attitude](images/attitude_tracking.png)

## The quadrotor tracking
![tracking](images/quad_mpc_tracking.png)
![control](images/quad_mpc_control.png)

## Todo

* Implement LNMPC controller for stable behavior