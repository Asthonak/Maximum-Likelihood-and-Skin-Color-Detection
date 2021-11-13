#include <iostream>
#include "Classifiers.cpp"
#include "boxmuller.cpp"
#include <math.h>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
	vector<Vector2f> distribution1;
	vector<Vector2f> distribution2;

	Vector2f mu_1;
	Matrix2f sigma_1;
	Vector2f mu_2;
	Matrix2f sigma_2;

	/*Problem 1*/

	// vectors/matricies to hold mu and sigma
	mu_1 << 1, 1;
	sigma_1 << 1, 0, 
		   0, 1;
	mu_2 << 4, 4;
	sigma_2 << 1, 0, 
		   0, 1;

	// generate the samples
	for(int i = 0; i < 100000; i++)
	{
		distribution1.push_back(Vector2f(box_muller(mu_1(0,0), sigma_1(0,0)), box_muller(mu_1(1,0), sigma_1(1,1))));
	}

	for(int i = 0; i < 100000; i++)
	{
		distribution2.push_back(Vector2f(box_muller(mu_2(0,0), sigma_2(0,0)), box_muller(mu_2(1,0), sigma_2(1,1))));
	}

	cout << "Reference for Problem 1: " << endl;

	// array to hold decision made for the classifer
	bool p1_reference[200000];

	// prior probabilities for reference and 1a
	float prior_prob_1 = 0.5;
	float prior_prob_2 = 0.5;

	// classify the samples using case 1
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_1(distribution1[i], mu_1, sigma_1(0,0), prior_prob_1);
			float gj = discriminant_case_1(distribution1[i], mu_2, sigma_2(0,0), prior_prob_2);

			if (gi - gj > 0)
			{
				p1_reference[i] = 1;
			}
			else
			{
				p1_reference[i] = 0;
			}

			gi = discriminant_case_1(distribution2[i], mu_1, sigma_1(0,0), prior_prob_1);
			gj = discriminant_case_1(distribution2[i], mu_2, sigma_2(0,0), prior_prob_2);

			if (gi - gj > 0)
			{
				p1_reference[j] = 1;
			}
			else
			{
				p1_reference[j] = 0;
			}
	}

	// calculate the number of misclassifications made
	int wrong1 = 0;
	int wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p1_reference[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_reference[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	cout << "Problem 1a: " << endl;

	// find estimated sample means
	Vector2f estimated_mu_1 = ml_mean(distribution1, 100000);
	Vector2f estimated_mu_2 = ml_mean(distribution2, 100000);

	// find estimated sample covariances
	Matrix2f estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 100000);
	Matrix2f estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 100000);

	cout << "Estimated mu_1 (using 100% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 100% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 100% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 100% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p1_a_decision[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_a_decision[i] = 1;
			}
			else
			{
				p1_a_decision[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_a_decision[j] = 1;
			}
			else
			{
				p1_a_decision[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p1_a_decision[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_a_decision[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	cout << "Problem 1b: " << endl;

	//shuffle distribution
	random_shuffle(distribution1.begin(), distribution1.end());

	// find estimated sample means using 0.01% of samples
	estimated_mu_1 = ml_mean(distribution1, 10);
	estimated_mu_2 = ml_mean(distribution2, 10);
	// find estimated sample covariances using 0.01% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 10);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 10);

	cout << "Estimated mu_1 (using 0.01% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 0.01% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 0.01% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 0.01% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p1_b_decision_1[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_1[i] = 1;
			}
			else
			{
				p1_b_decision_1[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_1[j] = 1;
			}
			else
			{
				p1_b_decision_1[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p1_b_decision_1[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_b_decision_1[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// find estimated sample means using 0.1% of samples
	estimated_mu_1 = ml_mean(distribution1, 100);
	estimated_mu_2 = ml_mean(distribution2, 100);
	// find estimated sample covariances using 0.1% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 100);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 100);

	cout << "Estimated mu_1 (using 0.1% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 0.1% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 0.1% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 0.1% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p1_b_decision_2[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_2[i] = 1;
			}
			else
			{
				p1_b_decision_2[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_2[j] = 1;
			}
			else
			{
				p1_b_decision_2[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p1_b_decision_2[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_b_decision_2[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// find estimated sample means using 1% of samples
	estimated_mu_1 = ml_mean(distribution1, 1000);
	estimated_mu_2 = ml_mean(distribution2, 1000);
	// find estimated sample covariances using 1% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 1000);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 1000);

	cout << "Estimated mu_1 (using 1% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 1% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 1% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 1% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p1_b_decision_3[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_3[i] = 1;
			}
			else
			{
				p1_b_decision_3[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_3[j] = 1;
			}
			else
			{
				p1_b_decision_3[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p1_b_decision_3[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_b_decision_3[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// find estimated sample means using 10% of samples
	estimated_mu_1 = ml_mean(distribution1, 10000);
	estimated_mu_2 = ml_mean(distribution2, 10000);
	// find estimated sample covariances using 10% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 10000);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 10000);

	cout << "Estimated mu_1 (using 10% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 10% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 10% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 10% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p1_b_decision_4[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_4[i] = 1;
			}
			else
			{
				p1_b_decision_4[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p1_b_decision_4[j] = 1;
			}
			else
			{
				p1_b_decision_4[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p1_b_decision_4[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_b_decision_4[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	/*Problem 2*/

	// vectors/matricies to hold mu and sigma
	mu_1 << 1, 1;
	sigma_1 << 1, 0, 
		   0, 1;
	mu_2 << 4, 4;
	sigma_2 << 4, 0, 
		   0, 8;

	distribution1.clear();
	distribution2.clear();

	// generate the samples
	for(int i = 0; i < 100000; i++)
	{
		distribution1.push_back(Vector2f(box_muller(mu_1(0,0), sigma_1(0,0)), box_muller(mu_1(1,0), sigma_1(1,1))));
	}

	for(int i = 0; i < 100000; i++)
	{
		distribution2.push_back(Vector2f(box_muller(mu_2(0,0), sigma_2(0,0)), box_muller(mu_2(1,0), sigma_2(1,1))));
	}

	cout << "Problem 2a: " << endl;

	// find estimated sample means using 100% of samples
	estimated_mu_1 = ml_mean(distribution1, 100000);
	estimated_mu_2 = ml_mean(distribution2, 100000);
	// find estimated sample covariances using 100% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 100000);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 100000);

	cout << "Estimated mu_1 (using 100% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 100% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 100% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 100% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p2_a_decision[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_a_decision[i] = 1;
			}
			else
			{
				p2_a_decision[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_a_decision[j] = 1;
			}
			else
			{
				p2_a_decision[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p2_a_decision[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p2_a_decision[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	cout << "Problem 2b: " << endl;

	// shuffle distribution
	random_shuffle(distribution1.begin(), distribution1.end());

	// find estimated sample means using 0.01% of samples
	estimated_mu_1 = ml_mean(distribution1, 10);
	estimated_mu_2 = ml_mean(distribution2, 10);
	// find estimated sample covariances using 0.01% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 10);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 10);

	cout << "Estimated mu_1 (using 0.01% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 0.01% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 0.01% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 0.01% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p2_b_decision_1[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_1[i] = 1;
			}
			else
			{
				p2_b_decision_1[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_1[j] = 1;
			}
			else
			{
				p2_b_decision_1[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p2_b_decision_1[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p2_b_decision_1[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// find estimated sample means using 0.1% of samples
	estimated_mu_1 = ml_mean(distribution1, 100);
	estimated_mu_2 = ml_mean(distribution2, 100);
	// find estimated sample covariances using 0.1% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 100);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 100);

	cout << "Estimated mu_1 (using 0.1% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 0.1% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 0.1% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 0.1% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p2_b_decision_2[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_2[i] = 1;
			}
			else
			{
				p2_b_decision_2[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_2[j] = 1;
			}
			else
			{
				p2_b_decision_2[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p2_b_decision_2[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p2_b_decision_2[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// find estimated sample means using 1% of samples
	estimated_mu_1 = ml_mean(distribution1, 1000);
	estimated_mu_2 = ml_mean(distribution2, 1000);
	// find estimated sample covariances using 1% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 1000);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 1000);

	cout << "Estimated mu_1 (using 1% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 1% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 1% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 1% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p2_b_decision_3[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_3[i] = 1;
			}
			else
			{
				p2_b_decision_3[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_3[j] = 1;
			}
			else
			{
				p2_b_decision_3[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p2_b_decision_3[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p2_b_decision_3[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// find estimated sample means using 10% of samples
	estimated_mu_1 = ml_mean(distribution1, 10000);
	estimated_mu_2 = ml_mean(distribution2, 10000);
	// find estimated sample covariances using 10% of samples
	estimated_sigma_1 = ml_covariance(distribution1, estimated_mu_1, 10000);
	estimated_sigma_2 = ml_covariance(distribution2, estimated_mu_2, 10000);

	cout << "Estimated mu_1 (using 10% of samples):" << endl;
	cout << estimated_mu_1 << endl;
	cout << "Estimated mu_2 (using 10% of samples):" << endl;
	cout << estimated_mu_2 << endl;
	cout << "Estimated sigma_1 (using 10% of samples):" << endl;
	cout << estimated_sigma_1 << endl;
	cout << "Estimated sigma_2 (using 10% of samples):" << endl;
	cout << estimated_sigma_2 << endl;

	bool p2_b_decision_4[200000];

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_4[i] = 1;
			}
			else
			{
				p2_b_decision_4[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], estimated_mu_1, estimated_sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], estimated_mu_2, estimated_sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision_4[j] = 1;
			}
			else
			{
				p2_b_decision_4[j] = 0;
			}
	}

	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		if (i < 100000)
		{
			if (p2_b_decision_4[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p2_b_decision_4[i] == 1)
			{
				wrong2++;
			}
		}
	}

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;
}
