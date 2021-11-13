#include <iostream>
#include <fstream>

#include "ReadImage.cpp"
#include "WriteImage.cpp"
#include "ReadImageHeader.cpp"
#include "image.h"
#include "image.cpp"

#include <bits/stdc++.h> 

#include "Classifiers.cpp"
#include <math.h>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

int main()
{
	RGB val;

	// image variables
	int T_1_M, T_1_N, T_1_Q;
	bool T_1_type;
	int R_1_M, R_1_N, R_1_Q;
	bool R_1_type;

	ofstream output;

	// read training image
	string s = "Data/Training_1.ppm";
	int string_size = s.length(); 
	char training1_string[string_size];
	strcpy(training1_string, s.c_str());
	readImageHeader(training1_string, T_1_N, T_1_M, T_1_Q, T_1_type);
	ImageType training1(T_1_N, T_1_M, T_1_Q);
	readImage(training1_string, training1);

	// read reference image
	s = "Data/ref1.ppm";
	string_size = s.length(); 
	char ref1_string[string_size];
	strcpy(ref1_string, s.c_str());
	readImageHeader(ref1_string, R_1_N, R_1_M, R_1_Q, R_1_type);
	ImageType ref1(R_1_N, R_1_M, R_1_Q);
	readImage(ref1_string, ref1);

	// vectors to store the sample data
	vector<Vector2f> skin_sample_data;
	vector<Vector2f> not_skin_sample_data;

	/***Problem 3 A***/
	cout << "Problem 3A:" << endl;

	// variables to stor estimated means and covariances for nomalized RGB values
	Vector2f skin_estimated_mu_RGB;
	Matrix2f skin_estimated_sigma_RGB;
	Vector2f not_skin_estimated_mu_RGB;
	Matrix2f not_skin_estimated_sigma_RGB;

	float total=0;
	float x1=0;
	float x2=0;

	// generate the skin and non-skin samples from first data set us normalized RGB values
	for(int i=0; i<T_1_N; i++)
	{
		for(int j=0; j<T_1_M; j++)
		{
			training1.getPixelVal(i, j, val);
			total = val.r + val.g + val.b;
			if(total == 0)
			{
				x1 = 0;
				x2 = 0;
			}
			else
			{
				x1 = (float)val.r / total;
				x2 = (float)val.g / total;
			}
			ref1.getPixelVal(i, j, val);
			if(val.r != 0 && val.g != 0 && val.b != 0)
			{
				skin_sample_data.push_back(Vector2f(x1, x2));
			}
			else
			{
				not_skin_sample_data.push_back(Vector2f(x1, x2));
			}
		}
	}

	// calculate estimated means and covariances
	skin_estimated_mu_RGB = ml_mean(skin_sample_data, skin_sample_data.size());
	skin_estimated_sigma_RGB = ml_covariance(skin_sample_data, skin_estimated_mu_RGB, skin_sample_data.size());
	not_skin_estimated_mu_RGB = ml_mean(not_skin_sample_data, not_skin_sample_data.size());
	not_skin_estimated_sigma_RGB = ml_covariance(not_skin_sample_data, not_skin_estimated_mu_RGB, not_skin_sample_data.size());

	/**Data Set 3 RGB**/
	cout <<  "Data Set 3 (RGB)" << endl;

	// image variables
	int T_3_M, T_3_N, T_3_Q;
	bool T_3_type;
	int R_3_M, R_3_N, R_3_Q;
	bool R_3_type;

	// read training image
	s = "Data/Training_3.ppm";
	string_size = s.length(); 
	char training3_string[string_size];
	strcpy(training3_string, s.c_str());
	readImageHeader(training3_string, T_3_N, T_3_M, T_3_Q, T_3_type);
	ImageType training3(T_3_N, T_3_M, T_3_Q);
	readImage(training3_string, training3);

	// read reference image
	s = "Data/ref3.ppm";
	string_size = s.length(); 
	char ref3_string[string_size];
	strcpy(ref3_string, s.c_str());
	readImageHeader(ref3_string, R_3_N, R_3_M, R_3_Q, R_3_type);
	ImageType ref3(R_3_N, R_3_M, R_3_Q);
	readImage(ref3_string, ref3);

	// variables to store the number of negatives, positives, false negatives, and false positives
	vector<float> falseNegatives, falsePositives;
	float negative, positive, false_negative, false_positive;
	total = 0;

	/**Find ROC plot points**/
	cout << "Threshold" << "\t" << "# False Pos" << "\t" 
	<< "# False Neg" << "\t"  << "# Postive" << "\t" << "# Negative" 
	<< "\t" << "fp/p" << "\t" << "\t" << "fn/n" << endl;
	for(float threshold = -5; threshold <= 0.00; threshold+=0.1)
	{
		negative = 0;
		positive = 0;
		false_negative = 0;
		false_positive = 0;
		for(int i=0; i<T_3_N; i++)
		{
			for(int j=0; j<T_3_M; j++)
			{
				training3.getPixelVal(i, j, val);
				total = val.r + val.g + val.b;
				x1 = (float)val.r / total;
				x2 = (float)val.g / total;

				bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_RGB, skin_estimated_sigma_RGB, threshold);

				ref3.getPixelVal(i, j, val);

				if(skin)
				{
					positive++;
				}
				else
				{
					negative++;
				}

				if((val.r != 0 && val.g != 0 && val.b != 0) && !skin)
				{
					false_negative++;
				}
				else if(!(val.r != 0 && val.g != 0 && val.b != 0) && skin)
				{
					false_positive++;
				}
			}
		}
		falseNegatives.push_back(false_negative);
		falsePositives.push_back(false_positive);
		cout << threshold << "\t" << "\t" << false_positive << "\t" << "\t" << 
		false_negative << "\t" << "\t" << positive << "\t" << "\t" << negative << 
		"\t" << false_positive/positive << "\t" << false_negative/negative << endl;
	}

	// log thresholds, false positives, and false negatives to output files
	output.open("Output/P3_a_data3_thresh.txt");
	for(float threshold = -5; threshold <= 0.01; threshold+=0.1)
	{
		output << threshold << endl;
	}
	output.close();

	output.open("Output/P3_a_data3_false_neg.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falseNegatives[i] << endl;
	}
	output.close();

	output.open("Output/P3_a_data3_false_pos.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falsePositives[i] << endl;
	}
	output.close();

	/**Create New Image**/ 
	// optimal threshold found via ROC plot
	total = 0;
	ImageType a3_output_image(T_3_N, T_3_M, T_3_Q);
	for(int i=0; i<T_3_N; i++)
	{
		for(int j=0; j<T_3_M; j++)
		{
			training3.getPixelVal(i, j, val);
			total = val.r + val.g + val.b;
			x1 = (float)val.r / total;
			x2 = (float)val.g / total;

			bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_RGB, skin_estimated_sigma_RGB, -0.9);

			if(skin)
			{
				a3_output_image.setPixelVal(i, j, RGB(val.r, val.g, val.b));
			}
			else
			{
				a3_output_image.setPixelVal(i, j, RGB(255, 255, 255));
			}
		}
	}

	// write image
	s = "Output/P3_a_image3_ouput.ppm";
	string_size = s.length(); 
	char a3_output[string_size];
	strcpy(a3_output, s.c_str());
	writeImage(a3_output, a3_output_image);

	/**Data Set 6 RGB**/
	cout <<  "Data Set 6 (RGB)" << endl;

	// image variables
	int T_6_M, T_6_N, T_6_Q;
	bool T_6_type;
	int R_6_M, R_6_N, R_6_Q;
	bool R_6_type;

	// read training image
	s = "Data/Training_6.ppm";
	string_size = s.length(); 
	char training6_string[string_size];
	strcpy(training6_string, s.c_str());
	readImageHeader(training6_string, T_6_N, T_6_M, T_6_Q, T_6_type);
	ImageType training6(T_6_N, T_6_M, T_6_Q);
	readImage(training6_string, training6);

	// read reference image
	s = "Data/ref6.ppm";
	string_size = s.length(); 
	char ref6_string[string_size];
	strcpy(ref6_string, s.c_str());
	readImageHeader(ref6_string, R_6_N, R_6_M, R_6_Q, R_6_type);
	ImageType ref6(R_6_N, R_6_M, R_6_Q);
	readImage(ref6_string, ref6);


	falseNegatives.clear();
	falsePositives.clear();
	total = 0;

	/**Find ROC plot points**/
	cout << "Threshold" << "\t" << "# False Pos" << "\t" 
	<< "# False Neg" << "\t"  << "# Postive" << "\t" << "# Negative" 
	<< "\t" << "fp/p" << "\t" << "\t" << "fn/n" << endl;
	for(float threshold = -5; threshold <= 0.00; threshold+=0.1)
	{
		negative = 0;
		positive = 0;
		false_negative = 0;
		false_positive = 0;
		for(int i=0; i<T_6_N; i++)
		{
			for(int j=0; j<T_6_M; j++)
			{
				training6.getPixelVal(i, j, val);
				total = val.r + val.g + val.b;
				x1 = (float)val.r / total;
				x2 = (float)val.g / total;

				bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_RGB, skin_estimated_sigma_RGB, threshold);

				ref6.getPixelVal(i, j, val);

				if(skin)
				{
					positive++;
				}
				else
				{
					negative++;
				}

				if((val.r != 0 && val.g != 0 && val.b != 0) && !skin)
				{
					false_negative++;
				}
				else if(!(val.r != 0 && val.g != 0 && val.b != 0) && skin)
				{
					false_positive++;
				}
			}
		}

		falseNegatives.push_back(false_negative);
		falsePositives.push_back(false_positive);
		cout << threshold << "\t" << "\t" << false_positive << "\t" << "\t" << 
		false_negative << "\t" << "\t" << positive << "\t" << "\t" << negative << 
		"\t" << false_positive/positive << "\t" << false_negative/negative << endl;
	}

	// log thresholds, false positives, and false negatives to output files
	output.open("Output/P3_a_data6_thresh.txt");
	for(float threshold = -5; threshold <= 0.01; threshold+=0.1)
	{
		output << threshold << endl;
	}
	output.close();

	output.open("Output/P3_a_data6_false_neg.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falseNegatives[i] << endl;
	}
	output.close();

	output.open("Output/P3_a_data6_false_pos.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falsePositives[i] << endl;
	}
	output.close();

	/**Create New Image**/ 
	// optimal threshold found via ROC plot
	total = 0;
	ImageType a6_output_image(T_6_N, T_6_M, T_6_Q);
	for(int i=0; i<T_6_N; i++)
	{
		for(int j=0; j<T_6_M; j++)
		{
			training6.getPixelVal(i, j, val);
			total = val.r + val.g + val.b;
			x1 = (float)val.r / total;
			x2 = (float)val.g / total;

			bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_RGB, skin_estimated_sigma_RGB, -1.1);

			if(skin)
			{
				a6_output_image.setPixelVal(i, j, RGB(val.r, val.g, val.b));
			}
			else
			{
				a6_output_image.setPixelVal(i, j, RGB(255, 255, 255));
			}
		}
	}

	// write image
	s = "Output/P3_a_image6_ouput.ppm";
	string_size = s.length(); 
	char a6_output[string_size];
	strcpy(a6_output, s.c_str());
	writeImage(a6_output, a6_output_image);

	/***Problem 3 B***/
	cout << "Problem 3B:" << endl;

	// variables to store estimated means and covariances for nomalized YCbCr values
	Vector2f skin_estimated_mu_YCC;
	Matrix2f skin_estimated_sigma_YCC;
	Vector2f not_skin_estimated_mu_YCC;
	Matrix2f not_skin_estimated_sigma_YCC;

	skin_sample_data.clear();
	not_skin_sample_data.clear();

	// generate the skin and non-skin samples from first data set us normalized YCbCr values
	for(int i=0; i<T_1_N; i++)
	{
		for(int j=0; j<T_1_M; j++)
		{
			training1.getPixelVal(i, j, val);
			x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b;
			x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;	
			ref1.getPixelVal(i, j, val);
			if(val.r != 0 && val.g != 0 && val.b != 0)
			{
				skin_sample_data.push_back(Vector2f(x1, x2));
			}
			else
			{
				not_skin_sample_data.push_back(Vector2f(x1, x2));
			}
		}
	}

	// calculate estimated means and covariances
	skin_estimated_mu_YCC = ml_mean(skin_sample_data, skin_sample_data.size());
	skin_estimated_sigma_YCC = ml_covariance(skin_sample_data, skin_estimated_mu_YCC, skin_sample_data.size());
	not_skin_estimated_mu_YCC = ml_mean(not_skin_sample_data, not_skin_sample_data.size());
	not_skin_estimated_sigma_YCC = ml_covariance(not_skin_sample_data, not_skin_estimated_mu_YCC, not_skin_sample_data.size());

	falseNegatives.clear();
	falsePositives.clear();

	/**Data Set 3 YCbCr**/
	cout <<  "Data Set 3 (YCbCr)" << endl;
	/**Find ROC plot points**/
	cout << "Threshold" << "\t" << "# False Pos" << "\t" 
	<< "# False Neg" << "\t"  << "# Postive" << "\t" << "# Negative" 
	<< "\t" << "fp/p" << "\t" << "\t" << "fn/n" << endl;
	for(float threshold = -5; threshold <= 0.00; threshold+=0.1)
	{
		negative = 0;
		positive = 0;
		false_negative = 0;
		false_positive = 0;
		for(int i=0; i<T_3_N; i++)
		{
			for(int j=0; j<T_3_M; j++)
			{
				training3.getPixelVal(i, j, val);
				x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b;
				x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;

				bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_YCC, skin_estimated_sigma_YCC, threshold);

				ref3.getPixelVal(i, j, val);

				if(skin)
				{
					positive++;
				}
				else
				{
					negative++;
				}

				if((val.r != 0 && val.g != 0 && val.b != 0) && !skin)
				{
					false_negative++;
				}
				else if(!(val.r != 0 && val.g != 0 && val.b != 0) && skin)
				{
					false_positive++;
				}
			}
		}

		falseNegatives.push_back(false_negative);
		falsePositives.push_back(false_positive);
		cout << threshold << "\t" << "\t" << false_positive << "\t" << "\t" << 
		false_negative << "\t" << "\t" << positive << "\t" << "\t" << negative << 
		"\t" << false_positive/positive << "\t" << false_negative/negative << endl;
	}

	// log thresholds, false positives, and false negatives to output files
	output.open("Output/P3_b_data3_thresh.txt");
	for(float threshold = -5; threshold <= 0.01; threshold+=0.1)
	{
		output << threshold << endl;
	}
	output.close();

	output.open("Output/P3_b_data3_false_neg.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falseNegatives[i] << endl;
	}
	output.close();

	output.open("Output/P3_b_data3_false_pos.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falsePositives[i] << endl;
	}
	output.close();

	/**Create New Image**/ 
	// optimal threshold found via ROC plot
	ImageType b3_output_image(T_3_N, T_3_M, T_3_Q);
	for(int i=0; i<T_3_N; i++)
	{
		for(int j=0; j<T_3_M; j++)
		{
			training3.getPixelVal(i, j, val);
			x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b;
			x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;
			bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_YCC, skin_estimated_sigma_YCC, -1.8);

			if(skin)
			{
				b3_output_image.setPixelVal(i, j, RGB(val.r, val.g, val.b));
			}
			else
			{
				b3_output_image.setPixelVal(i, j, RGB(255, 255, 255));
			}
		}
	}

	// write image
	s = "Output/P3_b_image3_ouput.ppm";
	string_size = s.length(); 
	char b3_output[string_size];
	strcpy(b3_output, s.c_str());
	writeImage(b3_output, b3_output_image);


	falseNegatives.clear();
	falsePositives.clear();

	/**Data Set 6 YCbCr**/
	cout <<  "Data Set 6 (YCbCr)" << endl;
	/**Find ROC plot points**/
	cout << "Threshold" << "\t" << "# False Pos" << "\t" 
	<< "# False Neg" << "\t"  << "# Postive" << "\t" << "# Negative" 
	<< "\t" << "fp/p" << "\t" << "\t" << "fn/n" << endl;
	for(float threshold = -5; threshold <= 0.00; threshold+=0.1)
	{
		negative = 0;
		positive = 0;
		false_negative = 0;
		false_positive = 0;
		for(int i=0; i<T_6_N; i++)
		{
			for(int j=0; j<T_6_M; j++)
			{
				training6.getPixelVal(i, j, val);
				x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b;
				x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;

				bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_YCC, skin_estimated_sigma_YCC, threshold);

				ref6.getPixelVal(i, j, val);

				if(skin)
				{
					positive++;
				}
				else
				{
					negative++;
				}

				if((val.r != 0 && val.g != 0 && val.b != 0) && !skin)
				{
					false_negative++;
				}
				else if(!(val.r != 0 && val.g != 0 && val.b != 0) && skin)
				{
					false_positive++;
				}
			}
		}

		falseNegatives.push_back(false_negative);
		falsePositives.push_back(false_positive);
		cout << threshold << "\t" << "\t" << false_positive << "\t" << "\t" << 
		false_negative << "\t" << "\t" << positive << "\t" << "\t" << negative << 
		"\t" << false_positive/positive << "\t" << false_negative/negative << endl;
	}

	// log thresholds, false positives, and false negatives to output files
	output.open("Output/P3_b_data6_thresh.txt");
	for(float threshold = -5; threshold <= 0.01; threshold+=0.1)
	{
		output << threshold << endl;
	}
	output.close();

	output.open("Output/P3_b_data6_false_neg.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falseNegatives[i] << endl;
	}
	output.close();

	output.open("Output/P3_b_data6_false_pos.txt");

	for(float threshold = -5, i = 0; threshold <= 0.01; threshold+=0.1, i++)
	{
		output << falsePositives[i] << endl;
	}
	output.close();

	/**Create New Image**/ 
	// optimal threshold found via ROC plot
	ImageType b6_output_image(T_6_N, T_6_M, T_6_Q);
	for(int i=0; i<T_6_N; i++)
	{
		for(int j=0; j<T_6_M; j++)
		{
			training6.getPixelVal(i, j, val);
			x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b;
			x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;
			bool skin = threshold_case_3(Vector2f(x1, x2), skin_estimated_mu_YCC, skin_estimated_sigma_YCC, -2.0);

			if(skin)
			{
				b6_output_image.setPixelVal(i, j, RGB(val.r, val.g, val.b));
			}
			else
			{
				b6_output_image.setPixelVal(i, j, RGB(255, 255, 255));
			}
		}
	}

	// write image
	s = "Output/P3_b_image6_ouput.ppm";
	string_size = s.length(); 
	char b6_output[string_size];
	strcpy(b6_output, s.c_str());
	writeImage(b6_output, b6_output_image);
	return (1);
}
