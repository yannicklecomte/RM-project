-------------- MOTOR CHARACTERIZATION --------------

This folder aims for the system identification of the motor-propeller system. This use data-driven method to identify al the coefficients of the second-order balance model

    \ddot{\theta} + \frac{b}{I} \dot{\theta} + \frac{K_{\theta}}{I}\theta = \frac{lK_t}{I}(\gamma_R - \gamma_L) + \frac{\tau_0}{I}

The coefficients are divided in two categories. The static coefficients are K_{\theta} and \tau_0. The dynamic coefficients are I and b. 
The coefficient K_{t} comes from the motor system identification.

-------------- Configuration --------------
- motor: ESC 30A grey - 2-3S LiPo BEC
- 1kg load cell with the HX711 module
- AMT11 incremental encoder

-------------- Acquisitions --------------
- 1_debug_motor_HARDWARE.py: It's for debugging purposes or to calibrate the ESC, using potentiometer to manually control each propeller
- 1.1_debug_all.py: For debug purposes
- 2_acquisition_static_charact.py: Acquire all the samples for the static characterization of the motor
- 2.1_post_processing_static_charact.py: Fit the samples of the static acquisition to find the static coefficients. Also, it plots all the graphs linked.
- 3_acquisition_dynamics_UP.py: Acquire step up response of the motor thrust.
- 3.1_post_processing_dynamics_UP.py: Post process the acquired campaign for the dynamic UP characterization, and fit all the normalized curves. Also, it plots all the graphs linked.
- 4_acquisition_dynamics_DOWN.py: Acquire step down response of the motor thrust.
- 4.1_post_processing_dynamics_DOWN.py: Post process the acquired campaign for the dynamic DOWN characterization, and fit all the normalized curves. Also, it plots all the graphs linked.

