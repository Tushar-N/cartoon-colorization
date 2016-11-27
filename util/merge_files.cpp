// <output executable> <output file> <input file 1> <input file 2> <input file 3> ... 

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <sstream>
using namespace std;
std::string to_string(int i)
{
    std::stringstream ss;
    ss << i;
    return ss.str();
}

int main(int argc, char* argv[])
{
	fstream fileWrite;
	fileWrite.open (argv[1], std::fstream::in | std::fstream::out | std::fstream::app);

	int count = 0;

	for(int i = 2; i<argc; i++)
	{
		int i_dec =0;
		fstream f;
		f.open(argv[i], fstream::in | fstream::out);

		std::string str;
		std::getline(f, str);
		
		std::string delimiter = ",";
		while (std::getline(f, str))
		{
			int pos = str.find(delimiter);
			std::string token = str.substr(0, pos);
			std::string residue = str.substr(pos, str.length());
			i_dec = atoi( token.c_str() );
			i_dec += count;
			string result = to_string(i_dec) + residue;
			fileWrite <<result<<"\n";
		}
		count = i_dec;
		f.close();
	}
	return 0;
}