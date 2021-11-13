#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>

using namespace Eigen;
using namespace std;

float squared(Vector2f x)
{
    return x.transpose() * x;
}

float discriminant_case_1 (Vector2f x, Vector2f mu, float sigma, float prior_prob)
{
	float discriminant = (((1.0/sigma) * mu).transpose() * x) + (-1.0 / (2*sigma)) * squared(mu);
	if(prior_prob == 0.5)
	{
		discriminant += log(prior_prob);
	}
	return discriminant;
}

float discriminant_case_3(Vector2f x, Vector2f mu, Matrix2f sigma, float prior_prob)
{
	float discriminant = (x.transpose() * (-0.5 * sigma.inverse()) * x) 
			   + ((sigma.inverse() * mu).transpose() * x)(0) + (-0.5 * mu.transpose() * sigma.inverse() * mu) 
			   + (-0.5 * log(sigma.determinant()));
	if(prior_prob == 0.5)
	{
		discriminant += log(prior_prob);
	}
	return discriminant;
}

float discriminant_min_distance(Vector2f x, Vector2f mu)
{
	return -1.0 * squared(x-mu);
}

Vector2f ml_mean(vector<Vector2f> x, int n)
{
	Vector2f mean;
	mean << 0, 0;
	for(int i=0;i<n;i++)
	{
		mean += x[i];
	}
	mean = mean / n;
	return mean;
}

Matrix2f ml_covariance(vector<Vector2f> x, Vector2f mean, int n)
{
	
	Matrix2f covariance;
	covariance << 0, 0, 
		      0, 0;
	for(int i=0;i<n;i++)
	{
		covariance = covariance + ((x[i]-mean) * (x[i]-mean).transpose());
	}
	covariance = covariance / n;
	return covariance;
}

bool threshold_case_3(Vector2f x, Vector2f mu, Matrix2f sigma, float threshold)
{
	return (-0.5 * (x - mu).transpose() * sigma.inverse() * (x - mu)) > threshold;
}
