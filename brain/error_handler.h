#pragma once

namespace brain {
enum ErrorHandlerType {
	ERR_HANDLER_ERROR,
	ERR_HANDLER_WARNING,
	ERR_HANDLER_SCRIPT,
	ERR_HANDLER_SHADER,
};

typedef void (*ErrorHandlerFunc)(
		void *,
		const char *,
		const char *,
		int p_line,
		const char *,
		const char *,
		ErrorHandlerType p_type);

struct ErrorHandlerList {

	ErrorHandlerFunc errfunc;
	void *userdata;

	ErrorHandlerList *next;

	ErrorHandlerList() {
		errfunc = 0;
		next = 0;
		userdata = 0;
	}
};

extern void add_error_handler(ErrorHandlerList *p_handler);
extern void remove_error_handler(ErrorHandlerList *p_handler);

} // namespace brain
