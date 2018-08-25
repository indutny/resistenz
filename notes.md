# Notes

1. Batch Norm needs `training=True`, and has to be added to control dependencies
   of the minimizer op
2. `same` padding for convolutions. Weird, but acceptable
3. Final layer should not have BN
4. Keeping dimensions right in the IoU (center vs size)
5. Correct `| cos (alpha - beta) |` multiplier for IoU (sign)
6. Sum of all our confidence losses instead of mean (mean doesn't work, and max
   doesn't work either)
7. Image normalization, `[ 0, 1 )`
8. Angle priors, size priors (needs investigation)
9. Anchor count! Does it make any difference?
10. IoU threshold interpreation (if it will work out)
