// Main application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const browseButton = document.getElementById('browse-button');
    const dropZone = document.getElementById('drop-zone');
    const fileList = document.getElementById('file-list');
    const extractButton = document.getElementById('extract-button');
    const backToUploadButton = document.getElementById('back-to-upload');
    const uploadStage = document.getElementById('upload-stage');
    const processingStage = document.getElementById('processing-stage');
    const resultsStage = document.getElementById('results-stage');
    const step1 = document.getElementById('step-1');
    const step2 = document.getElementById('step-2');
    const step3 = document.getElementById('step-3');
    const tableSearch = document.getElementById('table-search');
    const tableHeaders = document.getElementById('table-headers');
    const tableBody = document.getElementById('table-body');
    const recordCount = document.getElementById('record-count');
    const docsCount = document.getElementById('docs-count');
    const shipmentsCount = document.getElementById('shipments-count');
    const containersCount = document.getElementById('containers-count');
    const processingTime = document.getElementById('processing-time');
    const toggleColumnsBtn = document.getElementById('toggle-columns');
    const columnMenu = document.getElementById('column-menu');
    const columnList = document.getElementById('column-list');
    const chatToggle = document.getElementById('chat-toggle');
    const chatPanel = document.getElementById('chat-panel');
    const chatClose = document.getElementById('chat-close');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendMessage = document.getElementById('send-message');
    const exportExcel = document.getElementById('export-excel');
    const exportCsv = document.getElementById('export-csv');
    const exportJson = document.getElementById('export-json');
    const themeToggle = document.getElementById('theme-toggle');
    
    // Add data view tab functionality
    const dataViewTabs = document.querySelectorAll('.data-view-tab');
    const dataViews = document.querySelectorAll('.data-view');
    
    dataViewTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const viewName = this.dataset.view;
            
            // Update active tab
            dataViewTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Update active view
            dataViews.forEach(view => {
                view.classList.remove('active');
                if (view.id === viewName) {
                    view.classList.add('active');
                }
            });
        });
    });
    
    // Global variables
    let files = [];
    let extractedData = [];
    let columns = [];
    let startTime;
    let endTime;
    let draggedColumn = null;
    
    // Switch between stages
    function showStage(stageName) {
        uploadStage.classList.remove('active');
        processingStage.classList.remove('active');
        resultsStage.classList.remove('active');
        
        step1.classList.remove('active', 'completed');
        step2.classList.remove('active', 'completed');
        step3.classList.remove('active', 'completed');
        
        if (stageName === 'upload') {
            uploadStage.classList.add('active');
            step1.classList.add('active');
        } else if (stageName === 'processing') {
            processingStage.classList.add('active');
            step1.classList.add('completed');
            step2.classList.add('active');
        } else if (stageName === 'results') {
            resultsStage.classList.add('active');
            step1.classList.add('completed');
            step2.classList.add('completed');
            step3.classList.add('active');
        }
    }
    
    // File upload handling
    browseButton.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    // Drag and drop handling
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', function() {
        this.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    
    function handleFiles(fileList) {
        for (let i = 0; i < fileList.length; i++) {
            // Check if the file is already in the list
            if (!files.some(f => f.name === fileList[i].name)) {
                files.push(fileList[i]);
            }
        }
        
        updateFileList();
        updateExtractButton();
    }
    
    function updateFileList() {
        fileList.innerHTML = '';
        
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            const extension = file.name.split('.').pop().toLowerCase();
            let iconClass = 'fa-file';
            
            // Set icon based on file type
            if (['pdf'].includes(extension)) {
                iconClass = 'fa-file-pdf';
            } else if (['xls', 'xlsx', 'csv'].includes(extension)) {
                iconClass = 'fa-file-excel';
            } else if (['jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp'].includes(extension)) {
                iconClass = 'fa-file-image';
            }
            
            fileItem.innerHTML = `
                <div class="file-icon">
                    <i class="fas ${iconClass}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
                <div class="file-remove" data-index="${index}">
                    <i class="fas fa-times"></i>
                </div>
            `;
            
            fileList.appendChild(fileItem);
        });
        
        // Add remove event listeners
        document.querySelectorAll('.file-remove').forEach(button => {
            button.addEventListener('click', function() {
                const index = parseInt(this.dataset.index);
                files.splice(index, 1);
                updateFileList();
                updateExtractButton();
            });
        });
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function updateExtractButton() {
        if (files.length > 0) {
            extractButton.disabled = false;
        } else {
            extractButton.disabled = true;
        }
    }
    
    // Extract data from files
    extractButton.addEventListener('click', function() {
        startTime = new Date();
        showStage('processing');
        
        // Create form data to send
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files[]', file);
        });
        
        // Optional: Add any additional parameters
        formData.append('ez_id', ''); // You can add your own IDs here if needed
        
        // Send the request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            endTime = new Date();
            const processingDuration = (endTime - startTime) / 1000; // in seconds
            
            if (data.success) {
                extractedData = data.labeled_data || [];
                
                // Update stats
                docsCount.textContent = files.length;
                shipmentsCount.textContent = extractedData.length;
                
                // Count containers that have container_number
                const containerCount = extractedData.filter(item => 
                    item.container_number && item.container_number.trim() !== ''
                ).length;
                containersCount.textContent = containerCount;
                
                processingTime.textContent = processingDuration.toFixed(1) + 's';
                
                // Build table
                buildResultsTable(extractedData);
                
                // Build JSON viewer
                buildJSONViewer(extractedData);
                
                // Show results stage
                showStage('results');
                
                // Add AI welcome message
                setTimeout(() => {
                    addChatMessage(`I've analyzed your ${files.length} documents and found ${extractedData.length} shipping records. Ask me anything about this data!`, 'assistant');
                }, 1000);
            } else {
                // Show error
                showNotification(data.message || 'Error processing files', 'error');
                showStage('upload');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('An error occurred while processing your files', 'error');
            showStage('upload');
        });
    });
    
    // Build results table
    function buildResultsTable(data) {
        if (!data || data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="100%" class="text-center py-5">No data found</td></tr>';
            recordCount.textContent = '0 records';
            return;
        }
        
        // Define the exact column order based on the provided template
        columns = [
            'office_name', 
            'batch_no', 
            'customer', 
            'type', 
            'reference_number', 
            'booking_number', 
            'bol_number', 
            'po_number', 
            'container_number', 
            'container_size', 
            'container_type', 
            'pickup_location_name', 
            'delivery_location_name', 
            'delivery_street_address', 
            'delivery_city', 
            'delivery_state', 
            'delivery_zip', 
            'return_location', 
            'container_weight', 
            'commodity', 
            'number_of_packages', 
            'eta_date', 
            'steam_ship_line', 
            'vessel', 
            'voyage', 
            'cut_off_date', 
            'early_release_date', 
            'seal', 
            'pickup_number', 
            'pickup_appointment_date_time', 
            'delivery_appointment_date_time', 
            'Options', 
            'Tags', 
            'Notes'
        ];
        
        // Map fields that might have different names in the API response
        const fieldMapping = {
            'office': 'office_name',
            'notes': 'Notes'
        };
        
        // Build column visibility checkboxes
        buildColumnToggles(columns);
        
        // Build table headers
        tableHeaders.innerHTML = '';
        columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = formatColumnName(column);
            th.dataset.column = column;
            th.draggable = true; // Enable drag and drop for column reordering
            
            // Add drag and drop event listeners
            th.addEventListener('dragstart', handleDragStart);
            th.addEventListener('dragover', handleDragOver);
            th.addEventListener('drop', handleDrop);
            th.addEventListener('dragend', handleDragEnd);
            
            tableHeaders.appendChild(th);
        });
        
        // Build table body
        tableBody.innerHTML = '';
        data.forEach(item => {
            const tr = document.createElement('tr');
            
            columns.forEach(column => {
                const td = document.createElement('td');
                td.dataset.column = column;
                
                // Check for field mapping
                const mappedField = fieldMapping[column] || column;
                const alternateField = fieldMapping[mappedField] || mappedField;
                
                // Try to get the value from original field or mapped field
                let value = item[column] || item[mappedField] || item[alternateField] || '';
                
                // Format dates if needed
                if (['eta_date', 'cut_off_date', 'early_release_date'].includes(column) && value) {
                    value = formatDateNew(value);
                } else if (['pickup_appointment_date_time', 'delivery_appointment_date_time'].includes(column) && value) {
                    value = formatDate(value);
                }
                
                td.textContent = value || '';
                if (!value) {
                    td.innerHTML = '<span class="text-muted">-</span>';
                }
                
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
        
        // Update record count
        recordCount.textContent = data.length + (data.length === 1 ? ' record' : ' records');
    }
    
    // Format date in MM/DD/YYYY format
    function formatDateNew(dateString) {
        if (!dateString) return '';
        
        try {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return dateString; // If invalid date, return original
            
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const year = date.getFullYear();
            
            return `${month}/${day}/${year}`;
        } catch (e) {
            return dateString;
        }
    }
    
    // Format date and time in MM/DD/YYYY HH:MM format
    function formatDate(dateTimeString) {
        if (!dateTimeString) return '';
        
        try {
            const date = new Date(dateTimeString);
            if (isNaN(date.getTime())) return dateTimeString; // If invalid date, return original
            
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const year = date.getFullYear();
            const hours = String(date.getHours()).padStart(2, '0');
            const minutes = String(date.getMinutes()).padStart(2, '0');
            
            return `${month}/${day}/${year} ${hours}:${minutes}`;
        } catch (e) {
            return dateTimeString;
        }
    }
    
    // Column drag and drop handlers
    function handleDragStart(e) {
        this.classList.add('dragging');
        draggedColumn = this;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', this.dataset.column);
    }
    
    function handleDragOver(e) {
        e.preventDefault();
        this.classList.add('drag-over');
        return false;
    }
    
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        
        this.classList.remove('drag-over');
        
        const sourceColumn = draggedColumn.dataset.column;
        const targetColumn = this.dataset.column;
        
        if (sourceColumn !== targetColumn) {
            // Find positions of source and target in columns array
            const sourceIndex = columns.indexOf(sourceColumn);
            const targetIndex = columns.indexOf(targetColumn);
            
            // Reorder columns array
            columns.splice(sourceIndex, 1);
            columns.splice(targetIndex, 0, sourceColumn);
            
            // Rebuild table with new column order
            buildResultsTable(extractedData);
        }
        
        return false;
    }
    
    function handleDragEnd() {
        this.classList.remove('dragging');
        document.querySelectorAll('th').forEach(th => {
            th.classList.remove('drag-over');
        });
    }
    
    // Build column toggle checkboxes
    function buildColumnToggles(columns) {
        columnList.innerHTML = '';
        
        columns.forEach(column => {
            const item = document.createElement('div');
            item.className = 'column-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'column-checkbox';
            checkbox.id = `col-${column}`;
            checkbox.checked = true;
            checkbox.dataset.column = column;
            
            checkbox.addEventListener('change', function() {
                toggleColumnVisibility(column, this.checked);
            });
            
            const label = document.createElement('label');
            label.htmlFor = `col-${column}`;
            label.textContent = formatColumnName(column);
            
            item.appendChild(checkbox);
            item.appendChild(label);
            columnList.appendChild(item);
        });
        
        // Add column search functionality
        document.querySelector('.column-toggle-search').addEventListener('input', function() {
            const searchText = this.value.toLowerCase();
            
            document.querySelectorAll('.column-item').forEach(item => {
                const label = item.querySelector('label').textContent.toLowerCase();
                
                if (label.includes(searchText)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }
    
    // Toggle column visibility
    function toggleColumnVisibility(column, visible) {
        // Update table headers
        const th = tableHeaders.querySelector(`th[data-column="${column}"]`);
        if (th) {
            th.style.display = visible ? '' : 'none';
        }
        
        // Update table cells
        const cells = tableBody.querySelectorAll(`td[data-column="${column}"]`);
        cells.forEach(cell => {
            cell.style.display = visible ? '' : 'none';
        });
    }
    
    // Format column names for display
    function formatColumnName(column) {
        // Custom mappings for specific fields
        const fieldDisplayNames = {
            "office_name": "Office",
            "batch_no": "Batch no",
            "customer": "Customer",
            "type": "I/E",
            "reference_number": "Reference#",
            "booking_number": "Booking#",
            "bol_number": "BOL#",
            "po_number": "PO#",
            "container_number": "Container#",
            "container_size": "Size",
            "container_type": "Type",
            "pickup_location_name": "Pickup Location",
            "delivery_location_name": "Shipper/Consignee",
            "delivery_street_address": "Street",
            "delivery_city": "City",
            "delivery_state": "State",
            "delivery_zip": "Zip",
            "return_location": "Return Location",
            "container_weight": "Weight",
            "commodity": "Commodity",
            "number_of_packages": "#Pkgs",
            "eta_date": "ETA",
            "steam_ship_line": "SSL",
            "vessel": "Vessel",
            "voyage": "Voyage",
            "cut_off_date": "LFD/Cut-off",
            "early_release_date": "ERD",
            "seal": "Seal",
            "pickup_number": "Pickup#",
            "pickup_appointment_date_time": "Port/Rail Appt. Date/Time",
            "delivery_appointment_date_time": "Cust Appt. Date/Time",
            "Options": "Options",
            "Tags": "Tags",
            "Notes": "Quick Notes"
        };
        
        if (fieldDisplayNames[column]) {
            return fieldDisplayNames[column];
        }
        
        // Default formatting: capitalize words and replace underscores with spaces
        return column
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    // Toggle columns menu
    toggleColumnsBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        columnMenu.classList.toggle('show');
    });
    
    // Close columns menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!toggleColumnsBtn.contains(e.target) && !columnMenu.contains(e.target)) {
            columnMenu.classList.remove('show');
        }
    });
    
    // Back to upload button
    backToUploadButton.addEventListener('click', function() {
        showStage('upload');
    });
    
    // Table search functionality
    tableSearch.addEventListener('input', function() {
        const searchText = this.value.toLowerCase();
        
        const rows = tableBody.querySelectorAll('tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            let found = false;
            
            cells.forEach(cell => {
                if (cell.textContent.toLowerCase().includes(searchText)) {
                    found = true;
                }
            });
            
            row.style.display = found ? '' : 'none';
        });
        
        // Update visible record count
        const visibleRows = Array.from(tableBody.querySelectorAll('tr')).filter(row => row.style.display !== 'none').length;
        recordCount.textContent = visibleRows + (visibleRows === 1 ? ' record' : ' records');
    });
    
    // Chat panel toggle
    chatToggle.addEventListener('click', function() {
        chatPanel.style.display = chatPanel.style.display === 'flex' ? 'none' : 'flex';
    });
    
    chatClose.addEventListener('click', function() {
        chatPanel.style.display = 'none';
    });
    
    // Send chat message
    sendMessage.addEventListener('click', sendChatMessage);
    
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });
    
    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addChatMessage(message, 'user');
        chatInput.value = '';
        
        // Add loading indicator
        const loadingId = 'loading-' + Date.now();
        const loadingMessage = document.createElement('div');
        loadingMessage.className = 'chat-message assistant-message';
        loadingMessage.id = loadingId;
        loadingMessage.innerHTML = `
            <div class="d-flex align-items-center">
                <div class="spinner-border spinner-border-sm me-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span>Analyzing your request...</span>
            </div>
        `;
        chatMessages.appendChild(loadingMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Send question to server
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `question=${encodeURIComponent(message)}`
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            const loadingElement = document.getElementById(loadingId);
            if (loadingElement) {
                loadingElement.remove();
            }
            
            if (data.error) {
                addChatMessage(`I'm sorry, I encountered an error: ${data.error}`, 'assistant');
            } else {
                addChatMessage(data.answer, 'assistant');
            }
        })
        .catch(error => {
            // Remove loading message
            const loadingElement = document.getElementById(loadingId);
            if (loadingElement) {
                loadingElement.remove();
            }
            
            console.error('Error:', error);
            addChatMessage('I apologize, but I encountered an error while processing your question. Please try again.', 'assistant');
        });
    }
    
    function addChatMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        // Format message with paragraphs
        const formattedMessage = message
            .split('\n\n')
            .map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
            .join('');
        
        messageDiv.innerHTML = formattedMessage;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Export functions
    exportExcel.addEventListener('click', function(e) {
        e.preventDefault();
        exportTable('excel');
    });
    
    exportCsv.addEventListener('click', function(e) {
        e.preventDefault();
        exportTable('csv');
    });
    
    exportJson.addEventListener('click', function(e) {
        e.preventDefault();
        exportTable('json');
    });
    
    function exportTable(format) {
        // Get visible columns
        const visibleColumns = [];
        document.querySelectorAll('.column-checkbox').forEach(checkbox => {
            if (checkbox.checked) {
                visibleColumns.push(checkbox.dataset.column);
            }
        });
        
        // Get visible rows
        const visibleData = [];
        tableBody.querySelectorAll('tr').forEach(row => {
            if (row.style.display !== 'none') {
                const rowData = {};
                
                visibleColumns.forEach(column => {
                    const cell = row.querySelector(`td[data-column="${column}"]`);
                    if (cell) {
                        rowData[column] = cell.textContent.trim() === '-' ? '' : cell.textContent.trim();
                    } else {
                        rowData[column] = '';
                    }
                });
                
                visibleData.push(rowData);
            }
        });
        
        // Call export endpoint
        fetch('/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                format: format,
                data: visibleData
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (format === 'json') {
                    // For JSON, we can download directly
                    downloadFile(
                        JSON.stringify(visibleData, null, 2),
                        `shipping_data_${Date.now()}.json`,
                        'application/json'
                    );
                } else if (format === 'csv' && data.data) {
                    // Download CSV data
                    downloadFile(
                        data.data,
                        data.filename || `shipping_data_${Date.now()}.csv`,
                        'text/csv'
                    );
                } else {
                    // For other formats, show success message
                    showNotification(`${format.toUpperCase()} export successful!`, 'success');
                    
                    // If the server provided a download URL
                    if (data.downloadUrl) {
                        window.location.href = data.downloadUrl;
                    }
                }
            } else {
                showNotification(data.message || `Error exporting to ${format.toUpperCase()}`, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Even if the server fails, try to generate exports client-side
            if (format === 'csv') {
                exportToCSV(visibleData);
            } else if (format === 'json') {
                downloadFile(
                    JSON.stringify(visibleData, null, 2),
                    `shipping_data_${Date.now()}.json`,
                    'application/json'
                );
            } else {
                showNotification(`Error exporting to ${format.toUpperCase()}`, 'error');
            }
        });
    }
    
    function exportToCSV(data) {
        if (!data || !data.length) return;
        
        const columns = Object.keys(data[0]);
        let csv = columns.map(col => `"${formatColumnName(col)}"`).join(',') + '\n';
        
        data.forEach(row => {
            csv += columns.map(col => {
                // Escape quotes and wrap in quotes
                const value = row[col] || '';
                return `"${value.toString().replace(/"/g, '""')}"`;
            }).join(',') + '\n';
        });
        
        downloadFile(csv, `shipping_data_${Date.now()}.csv`, 'text/csv');
    }
    
    function downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
    }
    
    // Show notification
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.style.position = 'fixed';
        notification.style.bottom = '20px';
        notification.style.right = '20px';
        notification.style.padding = '15px 20px';
        notification.style.borderRadius = '8px';
        notification.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.1)';
        notification.style.zIndex = '9999';
        notification.style.display = 'flex';
        notification.style.alignItems = 'center';
        notification.style.transition = 'all 0.3s ease';
        notification.style.transform = 'translateY(20px)';
        notification.style.opacity = '0';
        
        // Add icon based on type
        let icon = 'info-circle';
        if (type === 'success') {
            icon = 'check-circle';
        } else if (type === 'error') {
            icon = 'exclamation-circle';
        }
        
        notification.innerHTML = `<i class="fas fa-${icon} me-2"></i> ${message}`;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateY(0)';
            notification.style.opacity = '1';
        }, 10);
        
        // Auto remove after delay
        setTimeout(() => {
            notification.style.transform = 'translateY(20px)';
            notification.style.opacity = '0';
            
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    // JSON Viewer functionality
    
    // Add theme toggle functionality for JSON viewer
    themeToggle.addEventListener('change', function() {
        const jsonViewerContainer = document.querySelector('.json-viewer-container');
        if (jsonViewerContainer) {
            jsonViewerContainer.classList.toggle('light-theme');
        }
    });
    
    // Build the JSON viewer
    function buildJSONViewer(data) {
        const jsonViewerContainer = document.getElementById('json-viewer');
        jsonViewerContainer.innerHTML = '';
        
        // If there's no data, show a message
        if (!data || data.length === 0) {
            jsonViewerContainer.innerHTML = '<div class="alert alert-info">No data available.</div>';
            return;
        }
        
        // Create the JSON viewer
        const jsonViewer = createJSONViewer(data);
        jsonViewerContainer.appendChild(jsonViewer);
        
        // Check theme toggle state
        if (themeToggle.checked) {
            jsonViewer.classList.add('light-theme');
        }
    }
    
    // Function to create a syntax-highlighted JSON viewer
    function createJSONViewer(jsonData) {
        // Create container
        const container = document.createElement('div');
        container.className = 'json-viewer-container';
        
        // Create toolbar
        const toolbar = document.createElement('div');
        toolbar.className = 'json-viewer-toolbar';
        
        // Add toolbar buttons
        const viewToggle = document.createElement('button');
        viewToggle.className = 'btn btn-sm btn-light me-2';
        viewToggle.innerHTML = '<i class="fas fa-code me-1"></i> Raw JSON';
        viewToggle.title = 'Toggle between formatted and raw JSON';
        
        const copyButton = document.createElement('button');
        copyButton.className = 'btn btn-sm btn-light me-2';
        copyButton.innerHTML = '<i class="fas fa-copy me-1"></i> Copy';
        copyButton.title = 'Copy JSON to clipboard';
        
        const downloadButton = document.createElement('button');
        downloadButton.className = 'btn btn-sm btn-light';
        downloadButton.innerHTML = '<i class="fas fa-download me-1"></i> Download';
        downloadButton.title = 'Download JSON file';
        
        toolbar.appendChild(viewToggle);
        toolbar.appendChild(copyButton);
        toolbar.appendChild(downloadButton);
        
        // Create pre and code elements
        const pre = document.createElement('pre');
        pre.className = 'json-viewer-pre';
        
        const code = document.createElement('code');
        code.className = 'json-code language-json';
        
        // Format JSON with syntax highlighting
        const formattedJSON = formatJSON(jsonData);
        code.innerHTML = formattedJSON;
        
        // Assemble the components
        pre.appendChild(code);
        container.appendChild(toolbar);
        container.appendChild(pre);
        
        // Add event listeners for toolbar buttons
        let isRawView = false;
        
        viewToggle.addEventListener('click', () => {
            isRawView = !isRawView;
            if (isRawView) {
                // Show raw JSON (no indentation)
                code.textContent = JSON.stringify(jsonData);
                viewToggle.innerHTML = '<i class="fas fa-indent me-1"></i> Formatted';
            } else {
                // Show formatted JSON
                code.innerHTML = formattedJSON;
                viewToggle.innerHTML = '<i class="fas fa-code me-1"></i> Raw JSON';
            }
        });
        
        copyButton.addEventListener('click', () => {
            const textToCopy = isRawView 
                ? JSON.stringify(jsonData)
                : JSON.stringify(jsonData, null, 2);
                
            navigator.clipboard.writeText(textToCopy)
                .then(() => {
                    showCopySuccessIndicator(copyButton);
                })
                .catch(err => {
                    console.error('Failed to copy text: ', err);
                    alert('Failed to copy to clipboard');
                });
        });
        
        downloadButton.addEventListener('click', () => {
            const jsonString = JSON.stringify(jsonData, null, 2);
            downloadFile(jsonString, `shipping_data_${Date.now()}.json`, 'application/json');
        });
        
        return container;
    }
    
    // Format JSON with syntax highlighting
    function formatJSON(jsonData) {
        const jsonString = JSON.stringify(jsonData, null, 2);
        
        // Simple syntax highlighting
        return jsonString
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, match => {
                let cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'json-key';
                        // Remove the colon from the matched string for styling
                        match = match.replace(/:$/, '') + ':';
                    } else {
                        cls = 'json-string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'json-boolean';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                // Return the match wrapped in a span with the appropriate class
                if (cls === 'json-key') {
                    return `<span class="${cls}">${match.replace(/:$/, '')}</span>:`;
                }
                return `<span class="${cls}">${match}</span>`;
            })
            .replace(/:\s+/g, ': ')  // Keep the colon and space together for styling
            .replace(/\n/g, '<br>')
            .replace(/\s{2}/g, '&nbsp;&nbsp;');  // Preserve indentation
    }
    
    // Show copy success indicator
    function showCopySuccessIndicator(button) {
        const originalHTML = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check me-1"></i> Copied!';
        button.disabled = true;
        
        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.disabled = false;
        }, 2000);
    }
});