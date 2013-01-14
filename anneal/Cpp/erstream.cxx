// erstream.c++		Implementation of general error reporting class

//                      (c) Copyright 1995, Everett F. Carter Jr.
//                      Permission is granted by the author to use
//			this software for any application provided this
//			copyright notice is preserved.


static const char rcsid[] = "@(#)erstream.c++	1.6 10:19:05 6/9/95   EFC";

#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <iostream>

#include "erstream.hpp"

// #define DEBUG

#ifdef DEBUG
#include <taygeta/traceback.hpp>
#endif

static const char SOH = 1;       // internally used to delimit each message
static const char WARN = 2;
static const char FATAL = 3;
static const char SILENT = 4;
static const char FAIL = 5;

long int ErrorStream::efirst = 0;
long int ErrorStream::elast = LONG_MAX - 1;
long int ErrorStream::msgnum = 0;

Errorbuf::~Errorbuf()
{
}

int Errorbuf::doallocate()
{
	return 0;
}


int Errorbuf::sync()
{
		return 0;
}


#ifdef INHERIT_ERRORBUF
int ErrorStream::overflow(int ch)
#else
int Errorbuf::overflow(int ch)
#endif
{
	return 0;
}

ErrorStream::ErrorStream(const char* title, std::ostream& user_os) : errstatus(0),
                 my_os(user_os), errcount(0)
{
}

ErrorStream::ErrorStream(const ErrorStream& er) : 
                 my_os(er.my_os)
{
}

ErrorStream::~ErrorStream()
{
}


// write out the error line
int ErrorStream::write_buf(const char* s,const int len, const int eof)
{
	return 1;	// return 1 for OK
}

void ErrorStream::close()
{
}          


// a nonmember function, mostly so I remember how to do a manipulator
std::ostream& terminate(std::ostream& os)
{
	return os;		// keeps the compiler happy
}

// increment the error status (post-increment)
#ifndef __ATT2__
#ifndef __GNUC__
ErrorStream ErrorStream::operator++(int)		   // DOES NOT WORK WITH GCC
{
	 ErrorStream retval(*this);
	 errstatus++;
         return retval;
}
#endif
#endif

void ErrorStream::nomore()
{
}

void ErrorStream::warning(const char *msg)
{
}

void ErrorStream::fail(const char *msg)
{
}

void ErrorStream::fatal(const char *msg)
{
}

void ErrorStream::memory(const void *p)
{
}


void ErrorStream::warning(const int eval, const char *msg)
{
}

void ErrorStream::fail(const int eval, const char *msg)
{
}

void ErrorStream::fatal(const int eval, const char *msg)
{
}

void ErrorStream::memory(const int eval, const void *p)
{
}

void Warning(ErrorStream& ehr, const int eval)
{
}

void Fail(ErrorStream& ehr, const int eval)
{
}

void Fatal(ErrorStream& ehr, const int eval)
{
}

ErrorStream& ErrorStream::operator=(const int err)
{
	flush();
	errstatus = err;
        return *this;
}

ErrorStream& ErrorStream::operator|=(const int err)
{
	flush();
	errstatus |= err;
        return *this;
}

