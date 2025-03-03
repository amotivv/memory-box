#!/bin/bash
# License application script for Memory Box
# This script adds AGPL license headers to files that do not already have them

# License header text for Python files
PYTHON_LICENSE='"""
Memory Box - A semantic memory storage and retrieval system
Copyright (C) 2025 amotivv, inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

'

# License header text for SQL files
SQL_LICENSE='--
-- Memory Box - A semantic memory storage and retrieval system
-- Copyright (C) 2025 amotivv, inc.
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU Affero General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU Affero General Public License for more details.
--
-- You should have received a copy of the GNU Affero General Public License
-- along with this program.  If not, see <https://www.gnu.org/licenses/>.
--

'

# Function to add license to file if it doesn't contain "GNU Affero"
add_license_if_needed() {
    local file=$1
    local license=$2

    if ! grep -q "GNU Affero" "$file"; then
        echo "Adding license to $file"
        
        # For Python files with shebang, insert after the shebang
        if [[ "$file" == *.py ]] && head -1 "$file" | grep -q "^#!"; then
            shebang=$(head -1 "$file")
            echo "Preserving shebang: $shebang"
            sed -i.bak "1d" "$file"  # Remove the shebang temporarily
            echo -e "$shebang\n$license$(cat $file)" > "$file"
        else
            # Otherwise, insert at the beginning
            echo -e "$license$(cat $file)" > "$file"
        fi
    else
        echo "License already exists in $file"
    fi
}

# Apply Python license headers
echo "Applying Python license headers..."
find . -type f -name "*.py" | while read file; do
    add_license_if_needed "$file" "$PYTHON_LICENSE"
done

# Apply SQL license headers
echo "Applying SQL license headers..."
find . -type f -name "*.sql" | while read file; do
    add_license_if_needed "$file" "$SQL_LICENSE"
done

echo "License application complete!"
