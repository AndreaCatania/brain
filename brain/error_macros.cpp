/*************************************************************************/
/*  error_macros.cpp                                                     */
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

#include "error_macros.h"

bool brain::_err_error_exists = false;
std::string brain::_last_error("");

static brain::ErrorHandlerList *error_handler_list = NULL;

void brain::_err_set_last_error(const char *p_err) {

	_last_error = p_err;
	_err_error_exists = true;
}

void brain::_err_clear_last_error() {

	_last_error.clear();
}

void brain::add_error_handler(ErrorHandlerList *p_handler) {

	p_handler->next = error_handler_list;
	error_handler_list = p_handler;
}

void brain::remove_error_handler(ErrorHandlerList *p_handler) {

	ErrorHandlerList *prev = NULL;
	ErrorHandlerList *l = error_handler_list;

	while (l) {

		if (l == p_handler) {

			if (prev)
				prev->next = l->next;
			else
				error_handler_list = l->next;
			break;
		}
		prev = l;
		l = l->next;
	}
}

void brain::_err_print_error(
		const char *p_function,
		const char *p_file,
		int p_line,
		const char *p_error,
		ErrorHandlerType p_type) {

	ErrorHandlerList *l = error_handler_list;
	while (l) {

		l->errfunc(
				l->userdata,
				p_function,
				p_file,
				p_line,
				p_error,
				_err_error_exists ? _last_error.c_str() : "",
				p_type);

		l = l->next;
	}

	if (_err_error_exists) {
		_last_error.clear();
		_err_error_exists = false;
	}
}

void brain::_err_print_index_error(const char *p_function, const char *p_file, int p_line, int64_t p_index, int64_t p_size, const char *p_index_str, const char *p_size_str, bool fatal) {

	std::string fstr(fatal ? "FATAL: " : "");
	std::string err(fstr + "Index " + p_index_str + "=" + brain::itos(p_index) + " out of size (" + p_size_str + "=" + brain::itos(p_size) + ")");
	brain::_err_print_error(p_function, p_file, p_line, err.c_str());
}
