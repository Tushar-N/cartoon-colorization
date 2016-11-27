// <output executable> <output file> <input folder> 

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <dirent.h>
#include <vector>
#include <algorithm>
using namespace std;

std::string to_string(int i)
{
    std::stringstream ss;
    ss << i;
    return ss.str();
}

int main(int argc, char* argv[])
{
	if(argc != 3){
		printf(" wrong format!!\n <output executable> <output file> <input folder>\n");
		return 0;
	}

	// Open output file
	fstream fileWrite;
	fileWrite.open (argv[1], std::fstream::in | std::fstream::out | std::fstream::app);

	// Read files within directory
	vector<string>filename;
	DIR *dir = NULL;
	struct dirent *cur = NULL;
	if ((dir = opendir (argv[2])) != NULL) {
  		while ((cur = readdir (dir)) != NULL) {
  			if (cur->d_name[0] != '.' && cur->d_name[strlen(cur->d_name)-1] != '~'){
    			filename.push_back(string(argv[2]) + cur->d_name);
    		}
  		}
  		closedir (dir);
  		sort(filename.begin(), filename.end());
	} 
	else {
  		/* could not open directory */
  		perror ("");
  		return EXIT_FAILURE;
	}

	int count = 0;
	// open file and read contents
	for(int i = 0; i<filename.size(); i++){

		int i_dec =0;
		fstream f;
		f.open(filename[i].c_str(), fstream::in | fstream::out);

		if(f.fail()){
			cout << filename[i] <<" fail!!";
		}

		// skip 1st line
		std::string str;
		std::getline(f, str);
		
		std::string delimiter = ",";
		while (std::getline(f, str)){

			int pos = str.find(delimiter);
			std::string token = str.substr(0, pos);
			std::string residue = str.substr(pos, str.length());
			i_dec = atoi( token.c_str() );
			i_dec += count;
			string result = to_string(i_dec) + residue;
			fileWrite << result <<"\n";
		}
		count = i_dec;
		f.close();
	} 
	return 0;
}