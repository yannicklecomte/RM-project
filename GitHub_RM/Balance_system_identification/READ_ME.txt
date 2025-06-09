-------------- BALANCE CHARACTERIZATION --------------

This folder aims for the system identification of the balance system. This use data-driven method to identify al the coefficients of the second-order balance model

    \ddot{\theta} + \frac{b}{I} \dot{\theta} + \frac{K_{\theta}}{I}\theta = \frac{lK_t}{I}(\gamma_R - \gamma_L) + \frac{\tau_0}{I}

The coefficients are divided in two categories. The static coefficients are K_{\theta} and \tau_0. The dynamic coefficients are I and b. 
The coefficient K_{t} comes from the motor system identification.

-------------- Configuration --------------
- left motor: ESC 30A grey - 2-3S LiPo BEC
- right motor: ESC 30A grey - 2-3S LiPo BEC
- distance from the fulcrum: 68cm
- other parameters have been changed but are detailed in the corresponding acquisition folders (like weight, wires position, …)

-------------- Acquisitions --------------
- 1_encoder_calib.py: this script makes the calibration of the encoder
- 2_balance_potentiometer_test_HARDWARE.py: It's for debugging purposes or to calibrate the ESC, using potentiometer to manually control each propeller
- 3_acquisition_static_charact.py: Acquire all the samples for the static characterization of the balance
- 3.1_post_processing.py: Fit the samples of the static acquisition to find the static coefficients. Also, it plots all the graphs linked.
- 4_acquisition_dynamic_charact.py: Acquire step up and down response of the balance's angle.
- 4.1_post_processing_dynamic_charact.py: Post process the acquired campaign for the dynamic characterization, and fit all the normalized curves. Also, it plots all the graphs linked.

-------------- Issues encountered --------------
1. Left motor turned-off sometimes because of bad quality ESC
	- not use the yellow ESC that are cheap but instead use the grey ones 

2. Motor start at different duty cycle values
	- due to the calibration of the ESC 
	- turn throttle to max, switch on power supply. Wait for two quick beeps, then quickly turn down the throttle to 0, where it should beep again. Afterwards, throttle can be used normally

3. Bearings are stiff with lot of friction
	- we removed the join that caused lot of friction, but less protection to dust (no so of a problem)

4. Problem of Hardware PWM generation
	- First, check with a LED if it is illuminating or not
	- Second, check the config file at /boot/firmware/config.txt" if the last line is well specified
	- Thirdly, check the cmd "pinctrl get" to see what's the channel of the specified GPIO pin

