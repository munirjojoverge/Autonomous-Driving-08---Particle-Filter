/**********************************************
* Self-Driving Car Nano-degree - Udacity
*  Created on: June 18, 2017
*      Author: Munir Jojo-Verge
**********************************************/

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#define DBL_EPSILON 0.00001

#include "particle_filter.h"

using namespace std;

// declare a random engine to be used across multiple and various method calls
random_device rd;
static default_random_engine randomGen(rd());

/*
***********************************************************************************
				INTRODUCTION TO PROBLEM
***********************************************************************************
% The state X of the robot is comprised of:
%   x = position coordinate x (lower case)
%   y = position coordinate y (lower case)
%   th = heading angle

% The State Space Model is:
% X[k+1] = A X[k] + B u[k] + Wk
% Y[k] = C x[k] + Vk
% Where:
%   A = state matrix
%   X[k] =  State vector at sample/time k (analogous for k+1)
%   B = input matrix
%   u[k] =  measured inputs, For simplicity this is 0. We are not actuating
%   Wk = unmeasured forces or faults. AKA System Noise
%   Y[k] = measurment at sample/time k. In theory, we should measure odometries from the wheel encoders and also the lidar and
%   from there calculate x, y and th (position of the vehicle) ..but for simplicity we will assume 3
%   readings, one for each...and therefore the observation is 3x1 vector
%   C = measurment matrix.
%   Vk = measurment noise.
*/

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;
  double weight_init = 1 / num_particles;
  
  // Create the particles and add random Sys / Model noise
  // This might not be computationally efficient but its done this way for conceptual readability.
	  for (int i = 0; i < num_particles; i++) {
		  vector<double> sys_noise = gen_gauss_noise(std);
		  Particle particle;
		  particle.id = i;
		  particle.x = x + sys_noise[0];
		  particle.y = y + sys_noise[1];
		  particle.theta = theta + sys_noise[2];
		  particle.weight = weight_init;

		  particles.push_back(particle);
	  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

	for (auto &p : particles) {

		/*****************************************************
		1) MOVE THE PARTICLE ACORDING TO OUR "BICYCLE" MODEL
		*****************************************************/

		double new_theta;
		// Calculate its new heading/yaw based on the yaw rate.		
		// Since we will be dividing by yaw_rate to calculate the particle position we need to handle "div by zero" first
		if (fabs(yaw_rate) < DBL_EPSILON) { // basically zero
			new_theta = p.theta; // Since yaw_rate is zero, the angle stays constant
			p.x += (velocity * cos(new_theta) * delta_t);
			p.y += (velocity * sin(new_theta) * delta_t);
		}
		else // we have yaw rate <> zero
		{
			// Calculate the new particle position assuming the "bicycle" model and also NO accelerations. All the un-measured foreces and faults will be consired Model Noise
			new_theta = p.theta + (yaw_rate * delta_t);
			p.x += (velocity / yaw_rate) * (sin(new_theta) - sin(p.theta));
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(new_theta));
		}
		p.theta = new_theta;

		/*****************************************************
		2) ADD THE MODEL/SYSTEM NOISE (UNMEASURED FORCES AND FAULTS)
		*****************************************************/
		vector<double> sys_noise = gen_gauss_noise(std_pos);

		p.x += sys_noise[0];
		p.y += sys_noise[1];
		p.theta += sys_noise[2];
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

	for (auto &p : particles) {

		// 1) TRANSFORMATION (simple 2D rotation matrix + translation)

		// This following vector represent the transformation of the observations vector w.r.t to the i - th particle.
		// What this really means is that this vector will hold the MAP coordenates of the observed/measured (lidar/Radar) landmarks if the particle was making the measurement
		// and that's the reason I called it particle_observations.
		std::vector<LandmarkObs> particle_observations;

		for (auto obs : observations) {
			LandmarkObs trans_obs;
			//Method 1
			trans_obs.x = p.x + (obs.x * cos(p.theta) - obs.y * sin(p.theta));
			trans_obs.y = p.y + (obs.x * sin(p.theta) + obs.y * cos(p.theta));

			particle_observations.push_back(trans_obs); // Again, these are the observations(measurements of landmark positions) w.r.t the particle p in MAP/World coordinates
		}

		// 2) ASSOCIATION. Now we are trying to find which landmark does EACH observation corresponds to. We will simply associate the closest one (closest neighbor)
		// This will actually give the mu(i) on the Weight Update Equation based on The Multivariate-Gaussian probability 
		// Let's calculate the CLOSEST LANDMARK TO EACH OBSERVATION


		// First and for efficiency I will go through all the landmarks and make a list of ONLY the ones that are in sensor range.
		// Then, after that, I will do the more "computantionally heavy" Eucleadian distance calculation using this list.
		//vector<Map::single_landmark_s> ObservationsLandmarksAssociations;
		dataAssociation2(particle_observations, sensor_range, map_landmarks, p.associations);
		

		/* 3) Calculating the Particle's Final Weight
		Now we that we have done the measurement transformations and associations, we have all the pieces we need to calculate
		the particle's final weight. The particles final weight will be calculated as the product of each measurement's Multivariate-Gaussian probability.
		The Multivariate-Gaussian probability has two dimensions, x and y.
		The "Xi" of the Multivariate-Gaussian is the i-th measurment (map coordenates): particle_ observations.
		The mean (mu-i) of the Multivariate-Gaussian is the measurement's associated landmark position (landmark associated with the Xi measurment - map coordinates)
		and the Multivariate-Gaussian's standard deviation (sigma) is described by our initial uncertainty in the x and y ranges.
		The Multivariate-Gaussian is evaluated at the point of the transformed measurement's position.
		*/		
		finalWeight(particle_observations, p, std_landmark, map_landmarks);
	}

	// Normilize the weights
	normilizeWeights(particles);
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	/*
	Resample if the effective number of particles is smaller than a threshold

	what the following code specifically does is randomly, uniformally, sample from
	the cummulative distribution of the probability distribution
	generated by the weighted vector P_w.  If you sample randomly over
	this distribution, you will select values based upon there statistical
	probability, and thus, on average, pick values with the higher weights
	(i.e. high probability of being correct given the observation z).
	store this new value to the new estimate which will go back into the
	next iteration
	*/

	// Update weights 
	std::vector<double> Pweights;	
	for (Particle p : particles) {
		Pweights.push_back(p.weight);		
	}	
	weights = Pweights;

	//Re-sample
	vector<Particle> new_particles;
	discrete_distribution<> distribution_weights(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		// pick up a random particle based on their weight. Heavier particles get picketup more often
		int selected_idx = distribution_weights(randomGen);
		new_particles.push_back(particles[selected_idx]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

vector<double> ParticleFilter::gen_gauss_noise(double std[]) {

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Generate the normal distributions for the noisy (either measurments or Syste model noise). 

	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	vector<double> returnVector;
	returnVector.push_back(dist_x(randomGen));
	returnVector.push_back(dist_y(randomGen));
	returnVector.push_back(dist_theta(randomGen));

	return returnVector;
}
void ParticleFilter::normilizeWeights(std::vector<Particle> &particles) {
	double WeightSum = 0.0;
	for (Particle p : particles)
	{
		WeightSum += p.weight;
	}

	for (Particle &p : particles)
	{
		p.weight /= WeightSum;
	}
}

void ParticleFilter::dataAssociation2(std::vector<LandmarkObs> particle_observations, double sensor_range, Map map_landmarks, std::vector<int> &associations) {
	
	vector<int> ParticleAssociations;

	for (auto p_obs : particle_observations) {

		vector<LandmarkObs> LandmarksWithinRange;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			// get idx (not id) and x,y coordinates
			double lm_x = map_landmarks.landmark_list[j].x_f;
			double lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_idx = j;// map_landmarks.landmark_list[j].id_i;

						   // only consider landmarks within sensor range of the particle 
						   // Rather than using the "dist" method considering a circular 
						   // region around the particle, I will consider a rectangular region that is computationally faster)
			if (fabs(lm_x - p_obs.x) <= sensor_range && fabs(lm_y - p_obs.y) <= sensor_range) {
				// add prediction to vector
				LandmarksWithinRange.push_back(LandmarkObs{ lm_idx, lm_x, lm_y });
			}
		}
		if (LandmarksWithinRange.size() == 0) {
			cout << "There are NO Landmarks within range! " << endl;
		}

		// Let's now find the closest "Euclidean" landmark to the observation just using the LandmarksWithinRange
		double clostest_dist = numeric_limits<double>::max();
		int closest_landmark_idx = 0; // In case there are NO landmarks within range, I want to have at least first index 

		for (LandmarkObs landmark_candidate : LandmarksWithinRange) {

			double x_dist = landmark_candidate.x - p_obs.x;
			double y_dist = landmark_candidate.y - p_obs.y;
			double dist = sqrt(x_dist*x_dist + y_dist*y_dist);

			if (dist < clostest_dist) {
				clostest_dist = dist;
				closest_landmark_idx = landmark_candidate.id; // This is where I stored the Map landmark Index
			}
		}
		ParticleAssociations.push_back(closest_landmark_idx);
	}
	associations =  ParticleAssociations;
}

void ParticleFilter::finalWeight(std::vector<LandmarkObs> particle_observations, Particle &p, double std_landmark[], Map map_landmarks) {
	/* compute bivariate-gaussian
	https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case

	In the 2 - dimensional nonsingular case (k = rank(Σ) = 2), the probability density function of a vector[X Y]′ is :
	f(x, y) =
	\frac{ 1 }{2 \pi  \sigma_X \sigma_Y \sqrt{ 1 - \rho ^ 2 }}
	\exp\left(
	-\frac{ 1 }{2(1 - \rho ^ 2)}\left[
	\frac{ (x - \mu_X) ^ 2 }{\sigma_X ^ 2} +
	\frac{ (y - \mu_Y) ^ 2 }{\sigma_Y ^ 2} -
	\frac{ 2\rho(x - \mu_X)(y - \mu_Y) }{\sigma_X \sigma_Y}
	\right]
	\right)\\
	\end{ align }
	where ρ (rho) is the correlation between X and Y. In our case is ZERO
	and where sigma_X>0 and sigma_Y>0
	*/
	double weight_product = 1;
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];
	if (sigma_x < DBL_EPSILON & sigma_y < DBL_EPSILON) {
		cerr << "Sigma X and/or Sigma Y is zero" << endl;
		p.weight = 0.0;
	}
	else
	{		
		double K1 = 1 / (2 * M_PI * sigma_x * sigma_y);

		for (int i = 0; i < particle_observations.size(); i++) {
			// Particle Observation "i-th" in MAP coordinates			
			double x = particle_observations[i].x;
			double y = particle_observations[i].y;
						
			// Predicted closest Landmark in MAP coordinates
			int Associated_landmark_Idx = p.associations[i];
			
			double mu_x = map_landmarks.landmark_list[Associated_landmark_Idx].x_f;
			double mu_y = map_landmarks.landmark_list[Associated_landmark_Idx].y_f;
			
			
			double gap_x = x - mu_x;
			double gap_y = y - mu_y;

			
			// Calculate the Probability density for this especific measurment/observation and it's predicted closest landmark
			double weight = K1 * exp(-0.5 * (pow(gap_x, 2) / pow(sigma_x, 2) + pow(gap_y, 2) / pow(sigma_y, 2)));
			
			//multiply densities for all measurements to calculate the likelyhood of this particle to be in the correct/real position.
			weight_product *= weight;
		}	
		p.weight = weight_product;		
	}

}