// erstream.hpp		Definition of an error stream error reporting class

//                      (c) Copyright 1995, Everett F. Carter Jr.
//                      Permission is granted by the author to use
//			this software for any application provided this
//			copyright notice is preserved.

// NOTE: post-increment not working with GCC

// static const char rcsid[] = "@(#)erstream.hpp	1.6 10:18:50 6/9/95   EFC";

#ifndef ERSTREAM_HPP_
#define ERSTREAM_HPP_ 1.6

#include <iostream>
#include <stdlib.h>

#ifndef __GNUC__
#ifndef INHERIT_ERRORBUF
#define INHERIT_ERRORBUF 0
#endif
#endif

#ifndef INHERIT_ERRORBUF
	class ErrorStream;
#endif

// the streambuf for the error handler, not used directly
class Errorbuf
{
	private:
#ifndef INHERIT_ERRORBUF
	  ErrorStream *ehp;           // NOT owned by the Errorbuf !!!
#endif
	  char *space;
	protected:
	  virtual int doallocate();
	public:
#ifdef INHERIT_ERRORBUF
	  Errorbuf() : space(NULL) {}
#else
	  Errorbuf() : ehp(NULL), space(NULL) {}
	  Errorbuf(ErrorStream* ehpl) : ehp(ehpl), space(NULL) {}
#endif
	 ~Errorbuf();

#ifndef INHERIT_ERRORBUF
	  void associate(ErrorStream* ehpl) { ehp = ehpl; }

  virtual int overflow(int c);

#endif
#ifdef __GNUC__
  virtual int underflow() { return 1; }
#endif

	  virtual int sync();


};


class ErrorStream;

struct ErrManipRec		// struct for single parameter manipulator
{


        int arg;
	void (*fp)(ErrorStream& ehr, const int a);

        ErrManipRec( void (*f)(ErrorStream& f, const int a ), const int a)
        			 : fp(f), arg(a) {}
};

// The actual error handler class
#ifdef INHERIT_ERRORBUF
class ErrorStream : public Errorbuf
#else
class ErrorStream : public std::ostream, virtual public std::ios
#endif
{

#ifndef INHERIT_ERRORBUF
#endif


	private:
	   static long int efirst;
	   static long int elast;
	   static long int msgnum;       // count of warning messages

           int errstatus;
	   int errcount;
  std::ostream& my_os;
	   char *prog;

	   void nomore();

	  friend void Warning(ErrorStream& ehr, const int eval);
	  friend void    Fail(ErrorStream& ehr, const int eval);
	  friend void   Fatal(ErrorStream& ehr, const int eval);

        protected:
           virtual int write_buf(const char* s, const int len, const int eof);   
#ifndef INHERIT_ERRORBUF
#endif                      
	  ErrorStream(const ErrorStream& er);
	public:
  ErrorStream(std::ostream& user_os = std::cerr) :  errcount(0),
#ifdef INHERIT_ERRORBUF
#else                                        
#endif
				my_os(user_os),prog(NULL),errstatus(0) { }
  ErrorStream(const char *s,std::ostream& user_os = std::cerr);
	  ~ErrorStream();

#ifdef INHERIT_ERRORBUF
#else
          virtual int sync() { return 0; }
#endif

	  void close();

	  void reset()			{ errstatus = 0;    }
	  int status() const		{ return errstatus; }
	  int count()  const            { return errcount;  }

	  static void first_message(const long int fmsg)        { efirst = fmsg-1; }
	  static void maximum_messages(const long int mxmsg)    { elast = mxmsg+1; }

	  // set the error status
	  ErrorStream& operator=(const int err);
          
          // add to the error status
          ErrorStream& operator|=(const int err);

	  // increment the error status (pre-increment)
          ErrorStream& operator++()		   
          			 { errstatus++;        return *this; }
#ifndef __ATT2__
	  // increment the error status (post-increment)
#ifndef __GNUC__
          ErrorStream operator++(int);		   // DOES NOT WORK WITH GCC
#endif
#endif
          ErrorStream& operator+=(const int inc)
          			 { errstatus += inc;   return *this; }
          // one way to write error messages
	  void warning(const char *msg = NULL);
	  void nonfatal(const char *msg = NULL) { warning(msg); }
	  void fatal(const char * = NULL );
	  void fail(const char *msg = NULL);
	  void memory(const void * = NULL );

	  void  warning(const int eval, const char *msg = NULL);
	  void nonfatal(const int eval, const char *msg = NULL)
         			 { warning(eval, msg); }
	  void    fatal(const int eval, const char *msg = NULL);
	  void     fail(const int eval, const char *msg = NULL);
	  void   memory(const int eval, const void * = NULL );


	  // maninpulator version of fatal and warning
          friend ErrorStream& fatal(ErrorStream& er)
          		    { er.fatal();   return er; }
          friend ErrorStream& warning(ErrorStream& er)
          		    { er.warning(); return er; } 
          friend ErrorStream& fail(ErrorStream& er)
          		    { er.fail(); return er; } 

          friend ErrManipRec   fatal(const int eval)
          		{ return ErrManipRec(0,eval);  }
          friend ErrManipRec warning(const int eval)
          		{ return ErrManipRec(0,eval); } 
          friend ErrManipRec fail(const int eval)
          		{ return ErrManipRec(0,eval); } 

	  typedef ErrorStream& (*ErrManip)(ErrorStream&);


	// applicator for the zero parameter manipulators
       friend ErrorStream& operator<<(ErrorStream& err, ErrManip f)
       				 { (*f)( err ); return err; }

	// applicator for the one parameter manipulators
       friend ErrorStream& operator<<(ErrorStream& err, ErrManipRec r)
       				 { r.fp(err, r.arg ); return err; }

};

// a nonmember functions, mostly so I remember how to do a manipulator
std::ostream& terminate(std::ostream& os);


#endif


