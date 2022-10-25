// SPOG
// Copyright (C) 2017-2021 SPOG Authors
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 
#include <SPOG-BFV/tool/version.h>

std::string GET_SPOGBFV_VERSION() {     
    std::ostringstream oss; 
    oss << SPOGBFV_VERSION_MAJOR << "." << SPOGBFV_VERSION_MINOR << "." << SPOGBFV_VERSION_PATCH; 
    if(SPOGBFV_VERSION_TWEAK != 0)
	    oss << " - " << SPOGBFV_VERSION_TWEAK;
    return oss.str();
}
