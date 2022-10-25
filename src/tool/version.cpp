/**
 * SPOG
 * Copyright (C) 2017-2019 SPOG Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <SPOG/tool/version.h>

std::string GET_SPOG_VERSION() {     
    std::ostringstream oss; 
    oss << SPOG_VERSION_MAJOR << "." << SPOG_VERSION_MINOR << "." << SPOG_VERSION_PATCH; 
    if(SPOG_VERSION_TWEAK != 0)
	    oss << " - " << SPOG_VERSION_TWEAK;
    return oss.str();
}
