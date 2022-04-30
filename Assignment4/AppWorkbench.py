from PyRT_Common import *
import matplotlib.pyplot as plt
from GaussianProcess import *

# ############################################################################################## #
# Given a list of hemispherical functions (function_list) and a set of sample positions over the #
#  hemisphere (sample_pos_), return the corresponding sample values. Each sample value results   #
#  from evaluating the product of all the functions in function_list for a particular sample     #
#  position.                                                                                     #
# ############################################################################################## #
def collect_samples(function_list, sample_pos_):
    sample_values = []
    for i in range(len(sample_pos_)):
        val = 1
        for j in range(len(function_list)):
            val *= function_list[j].eval(sample_pos_[i])
        sample_values.append(RGBColor(val, 0, 0))  # for convenience, we'll only use the red channel
    return sample_values


# ########################################################################################### #
# Given a set of sample values of an integrand, as well as their corresponding probabilities, #
# this function returns the classic Monte Carlo (cmc) estimate of the integral.               #
# ########################################################################################### #
def compute_estimate_cmc(sample_prob_, sample_values_):
    num_samples = len(sample_prob_)
    estimate = 0
    for (sample_prob, sample_value) in zip(sample_prob_, sample_values_):
        estimate += sample_value.r/sample_prob
    return estimate/num_samples



# ----------------------------- #
# ---- Main Script Section ---- #
# ----------------------------- #


# #################################################################### #
# STEP 0                                                               #
# Set-up the name of the used methods, and their marker (for plotting) #
# #################################################################### #
#methods_label = [('MC', 'o')]
methods_label = [('MC', 'o'), ('MC IS', 'v'), ('BMC', 'x'), ('BMC IS', '1')] # for later practices
n_methods = len(methods_label) # number of tested monte carlo methods

# ######################################################## #
#                   STEP 1                                 #
# Set up the function we wish to integrate                 #
# We will consider integrals of the form: L_i * brdf * cos #
# ######################################################## #
#l_i = ArchEnvMap()
l_i = Constant(1)
kd = 1
brdf = Constant(kd)
cosine_term = CosineLobe(3)
integrand = [l_i, brdf, cosine_term]  # l_i * brdf * cos

# ############################################ #
#                 STEP 2                       #
# Set-up the pdf used to sample the hemisphere #
# ############################################ #
uniform_pdf = UniformPDF()
#exponent = 1
#cosine_pdf = CosinePDF(exponent)


# ###################################################################### #
# Compute/set the ground truth value of the integral we want to estimate #
# NOTE: in practice, when computing an image, this value is unknown      #
# ###################################################################### #
ground_truth = cosine_term.get_integral()  # Assuming that L_i = 1 and BRDF = 1
print('Ground truth: ' + str(ground_truth))


# ################### #
#     STEP 3          #
# Experimental set-up #
# ################### #
ns_min = 20  # minimum number of samples (ns) used for the Monte Carlo estimate
ns_max = 101  # maximum number of samples (ns) used for the Monte Carlo estimate
ns_step = 20  # step for the number of samples
ns_vector = np.arange(start=ns_min, stop=ns_max, step=ns_step)  # the number of samples to use per estimate
n_estimates = 100  # the number of estimates to perform for each value in ns_vector
n_samples_count = len(ns_vector)

# Initialize a matrix of estimate error at zero
results = np.zeros((n_samples_count, n_methods))  # Matrix of average error


# ################################# #
#          MAIN LOOP                #
# ################################# #

def estimate_integral_abs_error_MC(num_samples, num_estimates, pdf):
    avg_abs_error = 0

    for _ in range(num_estimates):
        samples_pos, samples_prob = sample_set_hemisphere(num_samples, pdf)
        samples_value = collect_samples(integrand, samples_pos) 

        estimate = compute_estimate_cmc(samples_prob, samples_value)
        abs_error_estimate = abs(ground_truth - estimate)
        avg_abs_error += abs_error_estimate

    avg_abs_error /= num_estimates
    return avg_abs_error


gaussian_process = GP(SobolevCov(), Constant(1))

def estimate_integral_abs_error_BMC(num_samples, num_estimates, pdf):
    samples_pos, _ = sample_set_hemisphere(num_samples, pdf)
    samples_value = collect_samples(integrand, samples_pos) 
    gaussian_process.add_sample_pos(samples_pos)
    gaussian_process.add_sample_val(samples_value)

    avg_abs_error = 0

    for _ in range(num_estimates):
        estimate = gaussian_process.compute_integral_BMC().r
        abs_error_estimate = abs(ground_truth - estimate)
        avg_abs_error += abs_error_estimate

    avg_abs_error /= num_estimates
    return avg_abs_error


unknown_function = CosineLobe(3)
known_function = CosineLobe(1)

def estimate_integral_abs_error_MCIS(num_samples, num_estimates, pdf):
    avg_abs_error = 0

    for _ in range(num_estimates):
        sample_positions, sample_probs = sample_set_hemisphere(num_samples, pdf)
        sample_values = collect_samples([unknown_function, known_function], sample_positions)

        estimate = compute_estimate_cmc(sample_probs, sample_values)
        abs_error_estimate = abs(ground_truth - estimate)
        avg_abs_error += abs_error_estimate

    avg_abs_error /= num_estimates
    return avg_abs_error


gaussian_process_IS = GP(SobolevCov(), CosineLobe(1))

def estimate_integral_abs_error_BMCIS(num_samples, num_estimates, pdf):
    samples_pos, _ = sample_set_hemisphere(num_samples, pdf)
    samples_value = collect_samples(integrand, samples_pos) 
    gaussian_process_IS.add_sample_pos(samples_pos)
    gaussian_process_IS.add_sample_val(samples_value)

    avg_abs_error = 0

    for _ in range(num_estimates):
        estimate = gaussian_process_IS.compute_integral_BMC().r
        abs_error_estimate = abs(ground_truth - estimate)
        avg_abs_error += abs_error_estimate

    avg_abs_error /= num_estimates
    return avg_abs_error


methods = [estimate_integral_abs_error_MC, estimate_integral_abs_error_MCIS, estimate_integral_abs_error_BMC, 
    estimate_integral_abs_error_BMCIS]
pdfs = [uniform_pdf, CosinePDF(1), uniform_pdf, CosinePDF(1)]
num_estimates = [100, 100, 10, 10]

# for each sample count consideredz
for k, ns in enumerate(ns_vector):

    for (i, method_label) in enumerate(methods_label):
        method_name = method_label[0]
        method_function = methods[i]

        print(f'Computing estimates using {ns} samples and {method_name} method')

        abs_error_estimates = method_function(ns, num_estimates[i], pdfs[i])
        results[k, i] = abs_error_estimates


# ################################################################################################# #
# Create a plot with the average error for each method, as a function of the number of used samples #
# ################################################################################################# #
for k in range(len(methods_label)):
    method = methods_label[k]
    plt.plot(ns_vector, results[:, k], label=method[0], marker=method[1])

plt.legend()
plt.show()
