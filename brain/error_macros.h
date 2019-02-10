/*************************************************************************/
/*  error_macros.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#pragma once

#include "brain/error_handler.h"
#include "brain/string.h"
#include "brain/typedefs.h"

/**
  * Important this is a modified version of error_macros.h of Godot since
  * Don't use the OS class to track the errors and also it doesn't use mutex
  */

/**
 * Error macros. Unlike exceptions and asserts, these macros try to maintain consistency and stability
 * inside the code. It is recommended to always return processable data, so in case of an error, the
 * engine can stay working well.
 * In most cases, bugs and/or invalid data are not fatal and should never allow a perfectly running application
 * to fail or crash.
 */

/**
 * Pointer to the error macro priting function. Reassign to any function to have errors printed
 */

/** Function used by the error macros */

// function, file, line, error, explanation
namespace brain {

void _err_set_last_error(const char *p_err);
void _err_clear_last_error();

void _err_print_error(const char *p_function, const char *p_file, int p_line, const char *p_error, ErrorHandlerType p_type = ERR_HANDLER_ERROR);
void _err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, bool fatal = false);

/** An index has failed if m_index<0 or m_index >=m_size, the function exists */

extern bool _err_error_exists;
extern std::string _last_error;
} // namespace brain

#ifndef _STR
#define _STR(m_x) #m_x
#define _MKSTR(m_x) _STR(m_x)
#endif

#define _FNL __FILE__ ":"

#ifdef DEBUG_ENABLED
/** Print a warning string.
 */
#define ERR_EXPLAINC(m_reason)                \
	{                                         \
		brain::_err_set_last_error(m_reason); \
		brain::_err_error_exists = true;      \
	}
#define ERR_EXPLAIN(m_string)                                      \
	{                                                              \
		brain::_err_set_last_error(std::string(m_string).c_str()); \
		brain::_err_error_exists = true;                           \
	}

#define DEBUG_ONLY(code) code

#else

#define ERR_EXPLAIN(m_text)
#define ERR_EXPLAINC(m_text)
#define DEBUG_ONLY(code)

#endif

#ifdef __GNUC__
//#define FUNCTION_STR __PRETTY_FUNCTION__ - too annoying
#define FUNCTION_STR __FUNCTION__
#else
#define FUNCTION_STR __FUNCTION__
#endif

// Don't use this directly; instead, use any of the CRASH_* macros
#ifdef _MSC_VER
#define GENERATE_TRAP                       \
	__debugbreak();                         \
	/* Avoid warning about control paths */ \
	for (;;) {                              \
	}
#else
#define GENERATE_TRAP __builtin_trap();
#endif

// (*): See https://stackoverflow.com/questions/257418/do-while-0-what-is-it-good-for

#define ERR_FAIL_INDEX(m_index, m_size)                                                                                    \
	do {                                                                                                                   \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                            \
			brain::_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
			return;                                                                                                        \
		} else                                                                                                             \
			brain::_err_error_exists = false;                                                                              \
	} while (0); // (*)

/** An index has failed if m_index<0 or m_index >=m_size, the function exists.
* This function returns an error value, if returning Error, please select the most
* appropriate error condition from error_macros.h
*/

#define ERR_FAIL_INDEX_V(m_index, m_size, m_retval)                                                                        \
	do {                                                                                                                   \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                            \
			brain::_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size)); \
			return m_retval;                                                                                               \
		} else                                                                                                             \
			brain::_err_error_exists = false;                                                                              \
	} while (0); // (*)

/** Use this one if there is no sensible fallback, that is, the error is unrecoverable.
*   We'll return a null reference and try to keep running.
*/
#define CRASH_BAD_INDEX(m_index, m_size)                                                                                         \
	do {                                                                                                                         \
		if (unlikely((m_index) < 0 || (m_index) >= (m_size))) {                                                                  \
			brain::_err_print_index_error(FUNCTION_STR, __FILE__, __LINE__, m_index, m_size, _STR(m_index), _STR(m_size), true); \
			GENERATE_TRAP                                                                                                        \
		}                                                                                                                        \
	} while (0); // (*)

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
  * the function will exit.
  */

#define ERR_FAIL_NULL(m_param)                                                                                     \
	{                                                                                                              \
		if (unlikely(!m_param)) {                                                                                  \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter ' " _STR(m_param) " ' is null."); \
			return;                                                                                                \
		} else                                                                                                     \
			brain::_err_error_exists = false;                                                                      \
	}

#define ERR_FAIL_NULL_V(m_param, m_retval)                                                                         \
	{                                                                                                              \
		if (unlikely(!m_param)) {                                                                                  \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Parameter ' " _STR(m_param) " ' is null."); \
			return m_retval;                                                                                       \
		} else                                                                                                     \
			brain::_err_error_exists = false;                                                                      \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the function will exit.
 */

#define ERR_FAIL_COND(m_cond)                                                                                     \
	{                                                                                                             \
		if (unlikely(m_cond)) {                                                                                   \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true."); \
			return;                                                                                               \
		} else                                                                                                    \
			brain::_err_error_exists = false;                                                                     \
	}

/** Use this one if there is no sensible fallback, that is, the error is unrecoverable.
 */

#define CRASH_COND(m_cond)                                                                                               \
	{                                                                                                                    \
		if (unlikely(m_cond)) {                                                                                          \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Condition ' " _STR(m_cond) " ' is true."); \
			GENERATE_TRAP                                                                                                \
		}                                                                                                                \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the function will exit.
 * This function returns an error value, if returning Error, please select the most
 * appropriate error condition from error_macros.h
 */

#define ERR_FAIL_COND_V(m_cond, m_retval)                                                                                                   \
	{                                                                                                                                       \
		if (unlikely(m_cond)) {                                                                                                             \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. returned: " _STR(m_retval)); \
			return m_retval;                                                                                                                \
		} else                                                                                                                              \
			brain::_err_error_exists = false;                                                                                               \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the loop will skip to the next iteration.
 */

#define ERR_CONTINUE(m_cond)                                                                                                    \
	{                                                                                                                           \
		if (unlikely(m_cond)) {                                                                                                 \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. Continuing..:"); \
			continue;                                                                                                           \
		} else                                                                                                                  \
			brain::_err_error_exists = false;                                                                                   \
	}

/** An error condition happened (m_cond tested true) (WARNING this is the opposite as assert().
 * the loop will break
 */

#define ERR_BREAK(m_cond)                                                                                                     \
	{                                                                                                                         \
		if (unlikely(m_cond)) {                                                                                               \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. Breaking..:"); \
			break;                                                                                                            \
		} else                                                                                                                \
			brain::_err_error_exists = false;                                                                                 \
	}

/** Print an error string and return
 */

#define ERR_FAIL()                                                                            \
	{                                                                                         \
		brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method/Function Failed."); \
		brain::_err_error_exists = false;                                                     \
		return;                                                                               \
	}

/** Print an error string and return with value
 */

#define ERR_FAIL_V(m_value)                                                                                              \
	{                                                                                                                    \
		brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Method/Function Failed, returning: " __STR(m_value)); \
		brain::_err_error_exists = false;                                                                                \
		return m_value;                                                                                                  \
	}

/** Use this one if there is no sensible fallback, that is, the error is unrecoverable.
 */

#define CRASH_NOW()                                                                                  \
	{                                                                                                \
		brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: Method/Function Failed."); \
		GENERATE_TRAP                                                                                \
	}

/** Print an error string.
 */

#define ERR_PRINT(m_string)                                                  \
	{                                                                        \
		brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string); \
		brain::_err_error_exists = false;                                    \
	}

#define ERR_PRINTS(m_string)                                                                           \
	{                                                                                                  \
		brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, String(m_string).utf8().get_data()); \
		brain::_err_error_exists = false;                                                              \
	}

/** Print a warning string.
 */

#define WARN_PRINT(m_string)                                                                      \
	{                                                                                             \
		brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, m_string, ERR_HANDLER_WARNING); \
		brain::_err_error_exists = false;                                                         \
	}

#define WARN_PRINTS(m_string)                                                                                          \
	{                                                                                                                  \
		brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, std::string(m_string).c_str(), ERR_HANDLER_WARNING); \
		brain::_err_error_exists = false;                                                                              \
	}

#define WARN_DEPRECATED                                                                                                                                          \
	{                                                                                                                                                            \
		static volatile bool warning_shown = false;                                                                                                              \
		if (!warning_shown) {                                                                                                                                    \
			brain::_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "This method has been deprecated and will be removed in the future", ERR_HANDLER_WARNING); \
			brain::_err_error_exists = false;                                                                                                                    \
			warning_shown = true;                                                                                                                                \
		}                                                                                                                                                        \
	}
