-------------- SINGLE-FIDELITY BAYESIAN OPTIMIZATION --------------

This folder aims to run and post process the single-fidelity Bayesian optimization on the balance test bench
-------------- Scripts --------------

- 1_encoder_calib.py: For the calibration of the absolute encoder
- 1_balance_potnetiometer_HARDWARE.py: It's for debugging purposes or to calibrate the ESC, using potentiometer to manually control each propeller
- SFGP_BO_acquisition.py: Make the acquisition on the balance of a BO. All initial parameters can be selected.
- SFGP_BO_functions.py: It contains all the function needed by the acquisition script. In house built Gaussian Process Regression
- SFGP_BO_post_processing_SEQUENCES.py: Plot the time series, the normalized expected improvement integral at each iterations
- SFGP_BO_post_processing_SEQUENCES.py: (Re) Plot the posterior maps of mean, uncertainty and expected improvement.
