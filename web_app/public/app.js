// App state variables
let appState = {
  students: [],
  sessions: [],
  attendance: []
};

// Map tabs to titles and sub-titles
const tabMeta = {
  dashboard: { title: "แดชบอร์ดสด (Live Feed)", subtitle: "แสดงข้อมูลการเช็คชื่อแบบเรียลไทม์ผ่านกล้อง" },
  sessions: { title: "กำหนดเวลาเช็คชื่อ (Sessions)", subtitle: "จัดการคาบเรียนและช่วงเวลาเปิดรับสแกนใบหน้า" },
  report: { title: "รายงานการเข้าเรียน (Reports)", subtitle: "สรุปผลผู้เข้าเรียน สาย และขาดเรียนแบบเรียลไทม์" },
  register: { title: "ลงทะเบียนนักศึกษา (Register)", subtitle: "ลงทะเบียนใบหน้าคู่กับรหัสนักศึกษาลงระบบ AI" },
  users: { title: "จัดการบัญชีอาจารย์ (Users)", subtitle: "จัดการรายชื่อผู้ใช้อาจารย์ที่สามารถใช้งานระบบได้" },
  backup: { title: "สำรองและกู้ข้อมูลระบบ (Backup & Sync)", subtitle: "จัดการประวัติการเช็คชื่อ ความปลอดภัย และความสมบูรณ์" }
};

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', () => {
  const isLoggedIn = sessionStorage.getItem('isLoggedIn');
  if (isLoggedIn === 'true') {
    document.getElementById('login-overlay').classList.remove('active');
    updateSidebarByRole();
    fetchDatabase();
    initRealTimeAlerts();
  } else {
    document.getElementById('login-overlay').classList.add('active');
  }
  setupDragAndDrop();
  
  // Set default date for session form to today
  const today = new Date().toISOString().split('T')[0];
  document.getElementById('sess-date').value = today;
});

// Switch Tabs
function switchTab(tabName) {
  // Hide all sections
  document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
  // Show target section
  document.getElementById(`sect-${tabName}`).classList.add('active');
  
  // Deactivate all sidebar buttons
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  // Activate target button
  if (tabName === 'dashboard') document.getElementById('tab-dash').classList.add('active');
  else if (tabName === 'sessions') document.getElementById('tab-session').classList.add('active');
  else if (tabName === 'report') {
    document.getElementById('tab-report').classList.add('active');
    populateReportSessionsDropdown();
  }
  else if (tabName === 'register') document.getElementById('tab-reg').classList.add('active');
  else if (tabName === 'users') document.getElementById('tab-users').classList.add('active');
  else if (tabName === 'backup') document.getElementById('tab-backup').classList.add('active');

  // Update header text
  document.getElementById('page-title').innerText = tabMeta[tabName].title;
  document.getElementById('page-subtitle').innerText = tabMeta[tabName].subtitle;

  // Auto-close sidebar on mobile after choosing a tab
  const sidebar = document.querySelector('.sidebar');
  if (sidebar && sidebar.classList.contains('open')) {
    sidebar.classList.remove('open');
  }
}

// Toggle Sidebar on mobile
function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  if (sidebar) {
    sidebar.classList.toggle('open');
  }
}

// Fetch DB from API
async function fetchDatabase() {
  try {
    const res = await fetch(`/api/db?_=${Date.now()}`);
    if (!res.ok) throw new Error('Failed to fetch DB');
    const data = await res.json();
    appState = data;
    
    // Display server IP dynamically
    if (data.serverIp) {
      window.serverIp = data.serverIp;
      const statusSub = document.querySelector('.status-subtext');
      if (statusSub) {
        const currentProto = window.location.protocol;
        const currentPort = window.location.port ? `:${window.location.port}` : '';
        statusSub.innerHTML = `IP: <a href="${currentProto}//${data.serverIp}${currentPort}" target="_blank" style="color: var(--primary); font-weight: 600; text-decoration: none;">${currentProto}//${data.serverIp}${currentPort}</a>`;
      }
    }

    updateUI();
  } catch (err) {
    console.error('Error fetching database:', err);
  }
}

// Update the entire UI elements
function updateUI() {
  // Update stats counters
  document.getElementById('stat-registered-count').innerText = appState.students.length;
  
  // Count present students today
  const todayStr = new Date().toISOString().split('T')[0];
  const presentToday = new Set(
    appState.attendance
      .filter(a => a.time.startsWith(todayStr))
      .map(a => a.studentId)
  ).size;
  document.getElementById('stat-present-count').innerText = presentToday;

  // Count sessions today
  const sessionsToday = appState.sessions.filter(s => s.date === todayStr).length;
  document.getElementById('stat-session-count').innerText = sessionsToday;

  // Find active session
  const activeSession = appState.sessions.find(s => s.active);
  const banner = document.getElementById('active-session-banner');
  const pulse = document.getElementById('status-pulse');
  const pulseText = document.getElementById('status-text');

  if (activeSession) {
    const now = new Date();
    const offset = now.getTimezoneOffset();
    const localNow = new Date(now.getTime() - (offset * 60 * 1000));
    const localTodayStr = localNow.toISOString().split('T')[0];
    const currentTimeStr = localNow.toISOString().split('T')[1].substring(0, 5);

    if (activeSession.date !== localTodayStr || currentTimeStr < activeSession.startTime || currentTimeStr > activeSession.endTime) {
      banner.className = 'current-session-banner';
      banner.innerHTML = `<i class="fa-regular fa-bell pulse-bell"></i> <span>ไม่มีคาบเรียนที่กำลังเปิดเช็คชื่อ</span>`;
      
      pulse.className = 'pulse-dot orange';
      pulseText.innerText = 'ระบบสแตนด์บาย';
    } else {
      banner.className = 'current-session-banner active-session';
      banner.innerHTML = `<i class="fa-solid fa-circle-play text-green spinner"></i> <span>วิชา: <strong>[${activeSession.subjectCode || 'N/A'}] ${activeSession.subjectName || activeSession.className}</strong> (${activeSession.startTime} - ${activeSession.endTime})</span>`;
      pulse.className = 'pulse-dot green';
      pulseText.innerText = 'เปิดเช็คชื่อ (มีเซสชัน)';
    }
  } else {
    banner.className = 'current-session-banner';
    banner.innerHTML = `<i class="fa-regular fa-bell pulse-bell"></i> <span>ไม่มีคาบเรียนที่กำลังเปิดเช็คชื่อ</span>`;
    
    pulse.className = 'pulse-dot orange';
    pulseText.innerText = 'ระบบสแตนด์บาย';
  }

  // Populate Lists and Tables
  renderAttendanceTable();
  renderSessionsList();
  renderStudentsList();
  
  const userRole = sessionStorage.getItem('userRole');
  if (userRole === 'admin') {
    renderUsersTable();
  }
  
  updateBackupStatus(activeSession);
}

// Render recent attendance table
function renderAttendanceTable() {
  const tbody = document.getElementById('attendance-table-body');
  tbody.innerHTML = '';
  
  if (appState.attendance.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">ไม่มีรายการเช็คชื่อในระบบ</td></tr>`;
    return;
  }

  // Show newest first
  const sorted = [...appState.attendance].sort((a, b) => new Date(b.time) - new Date(a.time));

  sorted.forEach(a => {
    const row = document.createElement('tr');
    
    const timeFormatted = new Date(a.time).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    const dateFormatted = new Date(a.time).toLocaleDateString('th-TH', { day: 'numeric', month: 'short' });
    
    const statusHtml = a.status === 'late'
      ? `<span class="badge badge-orange"><i class="fa-solid fa-circle-exclamation"></i> มาสาย</span>`
      : `<span class="badge badge-green"><i class="fa-solid fa-circle-check"></i> ตรงเวลา</span>`;
      
    const modeHtml = a.offline 
      ? `<span class="badge badge-orange" title="บันทึกในเครื่องขณะเน็ตหลุด และส่งขึ้นเว็บแล้ว"><i class="fa-solid fa-wifi-slash"></i> Offline Sync</span>`
      : `<span class="badge badge-green" title="เช็คชื่อผ่านหน้าเว็บสดสำเร็จ"><i class="fa-solid fa-wifi"></i> Online</span>`;

    row.innerHTML = `
      <td><strong>${a.studentId}</strong></td>
      <td>${a.studentName}</td>
      <td>[${a.subjectCode || '-'}] ${a.className}</td>
      <td>${dateFormatted} ${timeFormatted} น.</td>
      <td>${statusHtml}</td>
      <td>${modeHtml}</td>
    `;
    tbody.appendChild(row);
  });
}

// Render sessions list
function renderSessionsList() {
  const container = document.getElementById('sessions-container');
  container.innerHTML = '';

  if (appState.sessions.length === 0) {
    container.innerHTML = `<p class="text-center text-muted">ไม่พบประวัติเซสชันเช็คชื่อ</p>`;
    return;
  }

  // Sort by date/time (newest first)
  const sorted = [...appState.sessions].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

  sorted.forEach(s => {
    const item = document.createElement('div');
    item.className = 'list-item session-item';
    
    const activeText = s.active ? 'กำลังเช็คชื่อ' : 'ปิดเช็คชื่อ';
    const activeIcon = s.active ? 'fa-solid fa-toggle-on text-green' : 'fa-solid fa-toggle-off';
    
    const locationText = s.location ? ` • <i class="fa-solid fa-location-dot"></i> ${s.location}` : '';
    const instructorText = s.instructorName ? ` • <i class="fa-solid fa-user-tie"></i> ${s.instructorName}` : '';
    const lateText = s.lateAfter ? ` • <i class="fa-solid fa-clock-rotate-left"></i> สายหลัง ${s.lateAfter}` : '';

    item.innerHTML = `
      <div class="session-item-header">
        <h4>[${s.subjectCode || 'N/A'}] ${s.subjectName || s.className}</h4>
        <span class="badge ${s.active ? 'badge-green' : 'badge-orange'}">${activeText}</span>
      </div>
      <div class="session-item-details">
        <span><i class="fa-regular fa-calendar"></i> วันที่: ${s.date}</span>
        <span><i class="fa-regular fa-clock"></i> เวลา: ${s.startTime} - ${s.endTime} น.${lateText}${locationText}${instructorText}</span>
      </div>
      <div class="session-item-actions">
        <button class="btn btn-secondary-outline btn-sm" onclick="toggleSession('${s.id}')">
          <i class="${activeIcon}"></i> ${s.active ? 'ปิดเช็คชื่อ' : 'เปิดเช็คชื่อ'}
        </button>
        <button class="btn btn-secondary-outline btn-sm" onclick="editSession('${s.id}')">
          <i class="fa-solid fa-pen-to-square"></i> แก้ไข
        </button>
        <button class="btn btn-danger-outline btn-sm" onclick="deleteSession('${s.id}')">
          <i class="fa-solid fa-trash"></i> ลบ
        </button>
      </div>
    `;
    container.appendChild(item);
  });
}

// Render registered students
function renderStudentsList() {
  const container = document.getElementById('students-container');
  container.innerHTML = '';

  if (appState.students.length === 0) {
    container.innerHTML = `<p class="text-center text-muted">ไม่พบข้อมูลนักศึกษาที่ลงทะเบียน</p>`;
    return;
  }

  // Sort alphabetically by ID
  const sorted = [...appState.students].sort((a, b) => a.id.localeCompare(b.id));

  sorted.forEach(s => {
    const item = document.createElement('div');
    item.className = 'list-item student-item';
    item.setAttribute('data-id', s.id);
    item.setAttribute('data-name', s.name.toLowerCase());
    
    const escapedName = s.name.replace(/'/g, "\\'").replace(/"/g, '&quot;');
    item.innerHTML = `
      <div class="student-item-info">
        <h4>${s.name}</h4>
        <p>ID: ${s.id}</p>
      </div>
      <div class="student-item-actions">
        <button class="btn btn-secondary-outline btn-xs" onclick="editStudent('${s.id}', '${escapedName}')">
          <i class="fa-solid fa-pen-to-square"></i> แก้ไข
        </button>
        <button class="btn btn-danger-outline btn-xs" onclick="deleteStudent('${s.id}')">
          <i class="fa-solid fa-trash"></i> ลบ
        </button>
      </div>
    `;
    container.appendChild(item);
  });
}

// Search student list (client-side filter)
function searchStudents() {
  const query = document.getElementById('student-search').value.toLowerCase().trim();
  const items = document.querySelectorAll('.student-item');
  
  items.forEach(item => {
    const id = item.getAttribute('data-id').toLowerCase();
    const name = item.getAttribute('data-name');
    if (id.includes(query) || name.includes(query)) {
      item.classList.remove('hide');
    } else {
      item.classList.add('hide');
    }
  });
}

// Variable to store editing session ID
let editingSessionId = null;

// Create or Update Session
async function createSession(e) {
  e.preventDefault();
  const subjectCode = document.getElementById('sess-subjectCode').value.trim();
  const subjectName = document.getElementById('sess-subjectName').value.trim();
  const location = document.getElementById('sess-location').value.trim();
  const instructorName = document.getElementById('sess-instructorName').value.trim();
  const date = document.getElementById('sess-date').value;
  const lateAfter = document.getElementById('sess-lateAfter').value;
  const startTime = document.getElementById('sess-start').value;
  const endTime = document.getElementById('sess-end').value;

  if (startTime >= endTime) {
    alert('เวลาเริ่มต้นต้องมาก่อนเวลาสิ้นสุด');
    return;
  }
  if (lateAfter && (lateAfter < startTime || lateAfter > endTime)) {
    alert('เวลาสายต้องอยู่ระหว่างเวลาเริ่มต้นและเวลาสิ้นสุด');
    return;
  }

  try {
    let url = '/api/sessions';
    let method = 'POST';
    
    if (editingSessionId) {
      url = `/api/sessions/${editingSessionId}`;
      method = 'PUT';
    }

    const res = await fetch(url, {
      method: method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subjectCode, subjectName, date, startTime, endTime, lateAfter, location, instructorName })
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || (editingSessionId ? 'แก้ไขข้อมูลคาบเรียนล้มเหลว' : 'สร้างคาบเรียนล้มเหลว'));
    }
    
    // Reset Form
    document.getElementById('sess-subjectCode').value = '';
    document.getElementById('sess-subjectName').value = '';
    document.getElementById('sess-location').value = '';
    document.getElementById('sess-instructorName').value = '';
    document.getElementById('sess-start').value = '';
    document.getElementById('sess-end').value = '';
    document.getElementById('sess-lateAfter').value = '';
    
    if (editingSessionId) {
      cancelEditSession();
      alert('แก้ไขข้อมูลคาบเรียนเสร็จสมบูรณ์!');
    } else {
      alert('บันทึกคาบเรียนและเปิดสแกนใบหน้าสำเร็จ!');
    }
    fetchDatabase();
  } catch (err) {
    alert('เกิดข้อผิดพลาด: ' + err.message);
  }
}

// Edit Session mode
function editSession(id) {
  const session = appState.sessions.find(s => s.id === id);
  if (!session) return;
  
  editingSessionId = id;
  
  // Fill inputs
  document.getElementById('sess-subjectCode').value = session.subjectCode || '';
  document.getElementById('sess-subjectName').value = session.subjectName || '';
  document.getElementById('sess-location').value = session.location || '';
  document.getElementById('sess-instructorName').value = session.instructorName || '';
  document.getElementById('sess-date').value = session.date || '';
  document.getElementById('sess-lateAfter').value = session.lateAfter || '';
  document.getElementById('sess-start').value = session.startTime || '';
  document.getElementById('sess-end').value = session.endTime || '';
  
  // Update UI Card Header
  const formCard = document.querySelector('#sect-sessions .card:first-child');
  formCard.querySelector('h2').innerHTML = '<i class="fa-solid fa-pen-to-square"></i> แก้ไขช่วงเวลาเช็คชื่อ';
  
  // Swap Submit Buttons
  const formActions = document.getElementById('session-form-actions');
  formActions.innerHTML = `
    <div style="display: flex; gap: 10px;">
      <button type="submit" class="btn btn-primary" style="flex: 1;"><i class="fa-solid fa-floppy-disk"></i> บันทึกการแก้ไข</button>
      <button type="button" class="btn btn-secondary-outline" onclick="cancelEditSession()"><i class="fa-solid fa-xmark"></i> ยกเลิก</button>
    </div>
  `;
  
  // Scroll to form (for mobile view comfort)
  document.getElementById('session-form').scrollIntoView({ behavior: 'smooth' });
}

// Cancel Edit Mode
function cancelEditSession() {
  editingSessionId = null;
  
  // Reset Form
  document.getElementById('session-form').reset();
  
  // Reset UI Card Header
  const formCard = document.querySelector('#sect-sessions .card:first-child');
  formCard.querySelector('h2').innerHTML = '<i class="fa-regular fa-clock"></i> ตั้งค่าช่วงเวลาเช็คชื่อใหม่';
  
  // Restore original Submit button
  const formActions = document.getElementById('session-form-actions');
  formActions.innerHTML = `
    <button type="submit" class="btn btn-primary btn-block"><i class="fa-solid fa-calendar-plus"></i> บันทึกและเปิดเซสชันนี้</button>
  `;
}

// Toggle Session active status
async function toggleSession(sessionId) {
  try {
    const res = await fetch('/api/sessions/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId })
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      alert(errData.error || 'ไม่สามารถเปิดเช็คชื่อได้เนื่องจากไม่อยู่ในช่วงเวลาเรียน');
      return;
    }
    fetchDatabase();
  } catch (err) {
    console.error('Error toggling session:', err);
    alert('เกิดข้อผิดพลาดในการเชื่อมต่อกับเซิร์ฟเวอร์');
  }
}

// Delete Session
async function deleteSession(id) {
  if (!confirm('คุณแน่ใจว่าต้องการลบคาบเรียนนี้? รายการเช็คชื่อทั้งหมดในคาบเรียนนี้จะถูกลบไปด้วย')) return;
  try {
    const res = await fetch(`/api/sessions/${id}`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error('Delete failed');
    fetchDatabase();
  } catch (err) {
    console.error('Error deleting session:', err);
  }
}

// Register Student (with file upload)
// Register Student (with multiple files upload support)
async function registerStudent(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-submit-register');
  const studentId = document.getElementById('reg-studentId').value.trim();
  const name = document.getElementById('reg-name').value.trim();
  const photoInput = document.getElementById('reg-photo');
  const alertBox = document.getElementById('reload-alert-box');

  if (!photoInput.files || photoInput.files.length === 0) {
    alert('กรุณาอัปโหลดรูปภาพใบหน้า (อย่างน้อย 1 รูป แนะนำ 5-10 รูปเพื่อความแม่นยำ)');
    return;
  }

  if (studentId.includes('-') || name.includes('-')) {
    alert('รหัสนักศึกษาและชื่อห้ามใส่เครื่องหมายลบ (-) เนื่องจากเป็นตัวแบ่งแยกชื่อของโฟลเดอร์ระบบ');
    return;
  }

  const formData = new FormData();
  formData.append('studentId', studentId);
  formData.append('name', name);
  
  // Append all selected photos
  for (let i = 0; i < photoInput.files.length; i++) {
    formData.append('photo', photoInput.files[i]);
  }

  // Disable button and show reloading banner
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner spinner"></i> กำลังอัปโหลดและประมวลผล...`;
  alertBox.classList.remove('hide');

  try {
    const res = await fetch('/api/students', {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || 'Registration failed');
    }
    
    // Reset form
    document.getElementById('register-form').reset();
    removeImage();
    
    setTimeout(() => {
      alertBox.classList.add('hide');
      btn.disabled = false;
      btn.innerHTML = `<i class="fa-solid fa-cloud-arrow-up"></i> บันทึกข้อมูลและประมวลผลใบหน้า`;
      fetchDatabase();
      alert(`ลงทะเบียนนักศึกษา ${name} เรียบร้อยแล้ว! ระบบ Python AI กำลังเทรนโมเดลใบหน้าใหม่ในเบื้องหลัง`);
    }, 1500);

  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-cloud-arrow-up"></i> บันทึกข้อมูลและประมวลผลใบหน้า`;
    alertBox.classList.add('hide');
    alert('บันทึกผิดพลาด: ' + err.message);
  }
}

// Image Previews gallery logic for multiple files
function previewFiles() {
  const photoInput = document.getElementById('reg-photo');
  const gallery = document.getElementById('image-preview-gallery');
  const previewContainer = document.getElementById('image-preview-container');
  const dropArea = document.getElementById('file-drop-area');
  const countLabel = document.getElementById('preview-count-label');

  gallery.innerHTML = '';
  const files = photoInput.files;

  if (files && files.length > 0) {
    countLabel.innerText = `เลือกแล้ว ${files.length} รูป (แนะนำ 5-10 รูป)`;
    previewContainer.classList.remove('hide');
    dropArea.classList.add('hide');

    Array.from(files).forEach(file => {
      const reader = new FileReader();
      reader.onload = function(e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.alt = 'Preview';
        img.style.width = '100%';
        img.style.height = '70px';
        img.style.objectFit = 'cover';
        img.style.borderRadius = 'var(--radius-sm)';
        img.style.border = '1px solid var(--glass-border)';
        gallery.appendChild(img);
      }
      reader.readAsDataURL(file);
    });
  }
}

function removeImage() {
  document.getElementById('reg-photo').value = '';
  document.getElementById('image-preview-gallery').innerHTML = '';
  document.getElementById('image-preview-container').classList.add('hide');
  document.getElementById('file-drop-area').classList.remove('hide');
}

// Setup drag and drop events
function setupDragAndDrop() {
  const dropArea = document.getElementById('file-drop-area');
  const fileInput = document.getElementById('reg-photo');

  ['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropArea.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropArea.classList.remove('drag-over');
    }, false);
  });

  dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      fileInput.files = files;
      previewFiles();
    }
  }, false);
}

// Backup check
function updateBackupStatus(activeSession) {
  const statusBadge = document.getElementById('client-conn-status');
  // Ping python port (5001) to see if it is running
  const hostname = window.location.hostname || 'localhost';
  fetch(`http://${hostname}:5001/status`, { method: 'GET', mode: 'no-cors', timeout: 1000 })
    .then(() => {
      statusBadge.innerText = 'เชื่อมต่อกล้องอยู่';
      statusBadge.className = 'badge badge-green';
    })
    .catch(() => {
      statusBadge.innerText = 'กล้องปิดอยู่ (Offline)';
      statusBadge.className = 'badge badge-red';
    });
}

// Real-Time Notification system using SSE
function initRealTimeAlerts() {
  const eventSource = new EventSource('/api/alerts');
  const container = document.getElementById('live-alert-container');

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const timeStr = new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    
    // Clear standby loader msg if it exists
    const standbyMsg = container.querySelector('.no-alerts-msg');
    if (standbyMsg) {
      container.innerHTML = '';
    }

    let alertHtml = '';
    let alertClass = '';

    switch(data.type) {
      case 'checkin_success':
        const alertStatus = data.payload.status || 'ontime';
        alertClass = alertStatus === 'late' ? 'alert-duplicate' : 'alert-success';
        const statusLabel = alertStatus === 'late' ? 'มาสาย' : 'ตรงเวลา';
        const iconHtml = alertStatus === 'late' ? '<i class="fa-solid fa-circle-exclamation"></i>' : '<i class="fa-solid fa-user-check"></i>';
        
        alertHtml = `
          <div class="alert-icon-circle">${iconHtml}</div>
          <div class="alert-content-info">
            <h4>เช็คชื่อสำเร็จ (${statusLabel}): ${data.payload.studentName}</h4>
            <p>รหัสนักศึกษา: ${data.payload.studentId} • วิชา: ${data.payload.className}</p>
          </div>
          <div class="alert-time-badge">${timeStr} น.</div>
        `;
        
        // Reload report in real-time if active in reports tab
        const courseSelect = document.getElementById('report-course-select');
        if (courseSelect && courseSelect.value === data.payload.subjectCode) {
          loadCourseReport();
        }
        
        // ดึงข้อมูลฐานข้อมูลมาอัปเดตสถิติและตารางบนหน้าเว็บแบบเรียลไทม์ทันที
        fetchDatabase();
        break;

      case 'checkin_duplicate':
        alertClass = 'alert-duplicate';
        alertHtml = `
          <div class="alert-icon-circle"><i class="fa-solid fa-triangle-exclamation"></i></div>
          <div class="alert-content-info">
            <h4>เช็คชื่อไปแล้ว: ${data.payload.studentName}</h4>
            <p>รหัส: ${data.payload.studentId} ยื่นตรวจซ้ำซ้อนในห้องเรียนนี้</p>
          </div>
          <div class="alert-time-badge">${timeStr} น.</div>
        `;
        break;

      case 'checkin_rejected':
        alertClass = 'alert-danger';
        alertHtml = `
          <div class="alert-icon-circle"><i class="fa-solid fa-circle-xmark"></i></div>
          <div class="alert-content-info">
            <h4>เช็คชื่อไม่ได้: ${data.payload.studentName || 'ไม่ทราบชื่อ'}</h4>
            <p>รหัส: ${data.payload.studentId || '-'} • เหตุผล: ${data.payload.reason}</p>
          </div>
          <div class="alert-time-badge">${timeStr} น.</div>
        `;
        break;

      case 'checkin_unknown':
        alertClass = 'alert-danger';
        alertHtml = `
          <div class="alert-icon-circle"><i class="fa-solid fa-user-slash"></i></div>
          <div class="alert-content-info">
            <h4>พบใบหน้าไม่รู้จักในกล้อง (Unknown Face)</h4>
            <p>ระบบไม่ได้ส่งข้อมูลเข้าคิวใบหน้า</p>
          </div>
          <div class="alert-time-badge">${timeStr} น.</div>
        `;
        break;

      case 'student_registered':
        alertClass = 'alert-success';
        alertHtml = `
          <div class="alert-icon-circle"><i class="fa-solid fa-user-plus text-blue"></i></div>
          <div class="alert-content-info">
            <h4>ลงทะเบียนใหม่สำเร็จ!</h4>
            <p>${data.payload.student.name} (รหัส ${data.payload.student.id})</p>
          </div>
          <div class="alert-time-badge">${timeStr} น.</div>
        `;
        fetchDatabase();
        break;

      case 'student_updated':
        alertClass = 'alert-success';
        alertHtml = `
          <div class="alert-icon-circle"><i class="fa-solid fa-user-pen text-blue"></i></div>
          <div class="alert-content-info">
            <h4>แก้ไขข้อมูลนักศึกษาสำเร็จ!</h4>
            <p>${data.payload.student.name} (รหัส ${data.payload.student.id})</p>
          </div>
          <div class="alert-time-badge">${timeStr} น.</div>
        `;
        fetchDatabase();
        break;

      case 'student_deleted':
        alertClass = 'alert-danger';
        alertHtml = `
          <div class="alert-icon-circle"><i class="fa-solid fa-user-minus text-red"></i></div>
          <div class="alert-content-info">
            <h4>ลบข้อมูลนักศึกษาแล้ว!</h4>
            <p>รหัสนักศึกษา: ${data.payload.studentId}</p>
          </div>
          <div class="alert-time-badge">${timeStr} น.</div>
        `;
        fetchDatabase();
        break;

      case 'session_change':
        fetchDatabase();
        return;
    }

    if (alertHtml) {
      const alertDiv = document.createElement('div');
      alertDiv.className = `alert-item ${alertClass}`;
      alertDiv.innerHTML = alertHtml;
      
      container.insertBefore(alertDiv, container.firstChild);
      
      while (container.children.length > 12) {
        container.removeChild(container.lastChild);
      }
      
      fetchDatabase();
    }
  };

  eventSource.onerror = (err) => {
    console.error("SSE Connection lost. Reconnecting...", err);
    document.getElementById('status-pulse').className = 'pulse-dot orange';
    document.getElementById('status-text').innerText = 'ขาดการติดต่อเว็บ';
  };
}

// Authentication handlers
async function handleLogin(e) {
  e.preventDefault();
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;
  const errBox = document.getElementById('login-error');

  errBox.classList.add('hide');

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'Login failed');
    }

    sessionStorage.setItem('isLoggedIn', 'true');
    sessionStorage.setItem('username', data.username);
    sessionStorage.setItem('userRole', data.role);
    
    document.getElementById('login-overlay').classList.remove('active');
    updateSidebarByRole();
    fetchDatabase();
    initRealTimeAlerts();
  } catch (err) {
    errBox.innerText = err.message;
    errBox.classList.remove('hide');
  }
}

function handleLogout() {
  sessionStorage.removeItem('isLoggedIn');
  sessionStorage.removeItem('username');
  sessionStorage.removeItem('userRole');
  document.getElementById('login-overlay').classList.add('active');
  // Clear inputs
  document.getElementById('login-username').value = '';
  document.getElementById('login-password').value = '';
}

// Sync students manually from local faces folder
async function syncStudentsFromFolder() {
  if (!confirm('คุณต้องการสแกนรายชื่อนักศึกษาทั้งหมดจากโฟลเดอร์ faces/ หรือไม่? (ระบบจะสแกนโฟลเดอร์รูปแบบ "รหัส-ชื่อ" และดึงเข้าระบบทันที)')) return;
  
  try {
    const res = await fetch('/api/students/sync', {
      method: 'POST'
    });
    if (!res.ok) throw new Error('Sync failed');
    const data = await res.json();
    appState.students = data.students;
    renderStudentsList();
    alert('ซิงค์รายชื่อจากโฟลเดอร์ใบหน้าสำเร็จ! ตรวจพบคลังนักศึกษาทั้งหมด: ' + data.students.length + ' คน');
  } catch (err) {
    alert('การซิงค์ผิดพลาด: ' + err.message);
  }
}

// Populate session select for detailed report
function populateReportSessionsDropdown() {
  const select = document.getElementById('report-session-select');
  const prevVal = select.value;
  select.innerHTML = '<option value="">-- เลือกคาบเรียน --</option>';
  
  const sorted = [...appState.sessions].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  sorted.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.innerText = `${s.date} | [${s.subjectCode || 'N/A'}] ${s.subjectName || s.className}`;
    select.appendChild(opt);
  });
  
  if (prevVal && [...select.options].some(o => o.value === prevVal)) {
    select.value = prevVal;
  }
}

// Load and render detailed session report
async function loadSessionReport() {
  const select = document.getElementById('report-session-select');
  const sessionId = select.value;
  const panel = document.getElementById('report-details-panel');
  const tbody = document.getElementById('report-table-body');
  
  if (!sessionId) {
    panel.classList.add('hide');
    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">กรุณาเลือกคาบเรียนจากแผงด้านข้าง</td></tr>';
    return;
  }
  
  try {
    const res = await fetch(`/api/sessions/${sessionId}/report`);
    if (!res.ok) throw new Error('Failed to load report data');
    const data = await res.json();
    const { session, report } = data;
    
    // Render session metadata
    document.getElementById('rep-subject').innerText = session.subjectName;
    document.getElementById('rep-code').innerText = 'รหัสวิชา: ' + (session.subjectCode || 'N/A');
    document.getElementById('rep-date').innerText = session.date;
    document.getElementById('rep-time').innerText = `${session.startTime} - ${session.endTime} น.`;
    document.getElementById('rep-late').innerText = session.lateAfter ? session.lateAfter + ' น.' : 'ไม่ระบุ';
    document.getElementById('rep-location').innerText = session.location || 'ไม่ระบุ';
    document.getElementById('rep-instructor').innerText = session.instructorName || 'ไม่ระบุ';
    
    // Render report table
    tbody.innerHTML = '';
    let ontimeCount = 0;
    let lateCount = 0;
    let absentCount = 0;
    
    if (report.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">ไม่มีข้อมูลนักศึกษาในระบบ</td></tr>';
    } else {
      const statusPriority = { ontime: 1, late: 2, absent: 3 };
      const sorted = [...report].sort((a, b) => {
        const priorityDiff = (statusPriority[a.status] || 99) - (statusPriority[b.status] || 99);
        if (priorityDiff !== 0) return priorityDiff;
        return a.studentId.localeCompare(b.studentId);
      });
      
      sorted.forEach(r => {
        const row = document.createElement('tr');
        let statusBadge = '';
        let scanMode = '-';
        let timeText = '-';
        
        if (r.status === 'ontime') {
          ontimeCount++;
          statusBadge = '<span class="badge badge-green"><i class="fa-solid fa-circle-check"></i> ตรงเวลา</span>';
          timeText = new Date(r.time).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) + ' น.';
          scanMode = r.offline 
            ? '<span class="badge badge-orange"><i class="fa-solid fa-wifi-slash"></i> Offline Sync</span>' 
            : '<span class="badge badge-green"><i class="fa-solid fa-wifi"></i> Online</span>';
        } else if (r.status === 'late') {
          lateCount++;
          statusBadge = '<span class="badge badge-orange"><i class="fa-solid fa-circle-exclamation"></i> มาสาย</span>';
          timeText = new Date(r.time).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) + ' น.';
          scanMode = r.offline 
            ? '<span class="badge badge-orange"><i class="fa-solid fa-wifi-slash"></i> Offline Sync</span>' 
            : '<span class="badge badge-green"><i class="fa-solid fa-wifi"></i> Online</span>';
        } else {
          absentCount++;
          statusBadge = '<span class="badge badge-red"><i class="fa-solid fa-circle-xmark"></i> ขาดเรียน</span>';
        }
        
        row.innerHTML = `
          <td><strong>${r.studentId}</strong></td>
          <td>${r.studentName}</td>
          <td>${timeText}</td>
          <td>${statusBadge}</td>
          <td>${scanMode}</td>
        `;
        tbody.appendChild(row);
      });
    }
    
    // Fill stats
    document.getElementById('rep-stat-ontime').innerText = ontimeCount;
    document.getElementById('rep-stat-late').innerText = lateCount;
    document.getElementById('rep-stat-absent').innerText = absentCount;
    
    // Store current report for exporting
    window.currentReport = { session, report };
    document.getElementById('btn-export-csv').style.display = 'inline-block';

    panel.classList.remove('hide');
  } catch (err) {
    console.error(err);
    document.getElementById('btn-export-csv').style.display = 'none';
    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-red">ดึงข้อมูลรายงานล้มเหลว: ' + err.message + '</td></tr>';
  }
}

// Export current session report to Excel-compatible CSV (UTF-8 with BOM)
function exportReportToCSV() {
  if (!window.currentReport || !window.currentReport.session || !window.currentReport.report) {
    alert('กรุณาเลือกคาบเรียนและโหลดข้อมูลรายงานก่อนทำการส่งออก');
    return;
  }
  
  const { session, report } = window.currentReport;
  
  // Create CSV Header and Info
  let csvContent = "\uFEFF"; // UTF-8 BOM to prevent Thai encoding issues in Excel
  csvContent += `รายงานการเข้าเรียนรายคาบ,,,\n`;
  csvContent += `วิชา,${session.subjectCode} - ${session.subjectName},,\n`;
  csvContent += `วันที่,${session.date},,\n`;
  csvContent += `เวลาเรียน,${session.startTime} - ${session.endTime} น.,,\n`;
  csvContent += `ผู้สอน,${session.instructorName || 'ไม่ระบุ'},,\n`;
  csvContent += `สถานที่,${session.location || 'ไม่ระบุ'},,\n`;
  csvContent += `\n`;
  csvContent += `รหัสนักศึกษา,ชื่อ-นามสกุล,เวลาเช็คชื่อ,สถานะ,รูปแบบการเช็คชื่อ\n`;
  
  report.forEach(r => {
    let statusText = 'ขาดเรียน';
    let timeText = '-';
    let modeText = '-';
    
    if (r.status === 'ontime') {
      statusText = 'ตรงเวลา';
      timeText = new Date(r.time).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) + ' น.';
      modeText = r.offline ? 'Offline Sync' : 'Online';
    } else if (r.status === 'late') {
      statusText = 'มาสาย';
      timeText = new Date(r.time).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) + ' น.';
      modeText = r.offline ? 'Offline Sync' : 'Online';
    }
    
    // Clean fields (wrap in quotes to handle commas)
    const id = `"${r.studentId.replace(/"/g, '""')}"`;
    const name = `"${r.studentName.replace(/"/g, '""')}"`;
    const time = `"${timeText.replace(/"/g, '""')}"`;
    const status = `"${statusText.replace(/"/g, '""')}"`;
    const mode = `"${modeText.replace(/"/g, '""')}"`;
    
    csvContent += `${id},${name},${time},${status},${mode}\n`;
  });
  
  // Download file
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  
  const sanitizedCode = (session.subjectCode || 'report').replace(/[^a-zA-Z0-9-]/g, '_');
  link.setAttribute("download", `${sanitizedCode}_Attendance_${session.date}.csv`);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Student Edit and Delete Handlers
function editStudent(id, name) {
  document.getElementById('edit-old-studentId').value = id;
  document.getElementById('edit-studentId').value = id;
  document.getElementById('edit-name').value = name;
  document.getElementById('edit-error').classList.add('hide');
  document.getElementById('edit-student-modal').classList.remove('hide');
}

function closeEditModal() {
  document.getElementById('edit-student-modal').classList.add('hide');
  document.getElementById('edit-error').classList.add('hide');
}

async function submitEditStudent(e) {
  e.preventDefault();
  const oldId = document.getElementById('edit-old-studentId').value;
  const newId = document.getElementById('edit-studentId').value.trim();
  const newName = document.getElementById('edit-name').value.trim();
  const errBox = document.getElementById('edit-error');

  if (!newId || !newName) {
    errBox.innerText = 'กรุณากรอกรหัสนักศึกษาและชื่อ-นามสกุล';
    errBox.classList.remove('hide');
    return;
  }

  try {
    const res = await fetch(`/api/students/${oldId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ newId, newName })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'เกิดข้อผิดพลาดในการแก้ไขข้อมูล');
    }

    // Update in local state
    appState.students = appState.students.map(s => {
      if (s.id === oldId) {
        return { ...s, id: newId, name: newName };
      }
      return s;
    });

    // Update attendance locally if any matched
    appState.attendance = appState.attendance.map(a => {
      if (a.studentId === oldId) {
        return { ...a, studentId: newId, studentName: newName };
      }
      return a;
    });

    renderStudentsList();
    closeEditModal();
    alert('แก้ไขข้อมูลนักศึกษาสำเร็จแล้ว ระบบกำลังทำการอัปเดตโมเดลใบหน้าใหม่');
  } catch (err) {
    errBox.innerText = err.message;
    errBox.classList.remove('hide');
  }
}

async function deleteStudent(id) {
  if (!confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบข้อมูลนักศึกษารหัส ${id} ?\n\nการดำเนินการนี้จะลบโฟลเดอร์รูปภาพใน faces/ และประวัติการเช็คชื่อทั้งหมดของนักศึกษาคนนี้ด้วย`)) {
    return;
  }

  try {
    const res = await fetch(`/api/students/${id}`, {
      method: 'DELETE'
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'เกิดข้อผิดพลาดในการลบข้อมูล');
    }

    // Remove from local state
    appState.students = appState.students.filter(s => s.id !== id);
    appState.attendance = appState.attendance.filter(a => a.studentId !== id);

    renderStudentsList();
    alert('ลบข้อมูลนักศึกษาสำเร็จแล้ว ระบบกำลังทำการอัปเดตโมเดลใบหน้าใหม่');
  } catch (err) {
    alert('เกิดข้อผิดพลาด: ' + err.message);
  }
}

// Update sidebar and UI based on user role
function updateSidebarByRole() {
  const role = sessionStorage.getItem('userRole') || 'teacher';
  const username = sessionStorage.getItem('username') || '';
  
  const usernameEl = document.getElementById('logged-in-username');
  if (usernameEl) {
    usernameEl.innerText = username;
  }

  const tabBackup = document.getElementById('tab-backup');
  const tabUsers = document.getElementById('tab-users');
  
  if (role === 'teacher') {
    if (tabBackup) tabBackup.style.display = 'none';
    if (tabUsers) tabUsers.style.display = 'none';
    const activeContent = document.querySelector('.tab-content.active');
    if (activeContent && (activeContent.id === 'sect-backup' || activeContent.id === 'sect-users')) {
      switchTab('dashboard');
    }
  } else {
    if (tabBackup) tabBackup.style.display = 'block';
    if (tabUsers) tabUsers.style.display = 'block';
  }
}

// Populate session select for detailed report
function populateReportSessionsDropdown() {
  const select = document.getElementById('report-session-select');
  if (!select) return;
  const prevVal = select.value;
  select.innerHTML = '<option value="">-- เลือกคาบเรียน --</option>';
  
  const sorted = [...appState.sessions].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  sorted.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.innerText = `${s.date} | [${s.subjectCode || 'N/A'}] ${s.subjectName || s.className}`;
    select.appendChild(opt);
  });
  
  if (prevVal && [...select.options].some(o => o.value === prevVal)) {
    select.value = prevVal;
  }
}

// Load and render detailed session report
async function loadSessionReport() {
  const select = document.getElementById('report-session-select');
  if (!select) return;
  const sessionId = select.value;
  const panel = document.getElementById('report-details-panel');
  const tbody = document.getElementById('course-report-table-body');
  
  if (!sessionId) {
    panel.classList.add('hide');
    if (tbody) tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">กรุณาเลือกคาบเรียนจากแผงด้านข้าง</td></tr>';
    return;
  }
  
  try {
    const session = appState.sessions.find(s => s.id === sessionId);
    if (!session) throw new Error('ไม่พบข้อมูลคาบเรียน');
    
    // Set metadata on UI
    document.getElementById('rep-subject').innerText = session.subjectName || session.className;
    document.getElementById('rep-code').innerText = 'รหัสวิชา: ' + (session.subjectCode || 'N/A');
    document.getElementById('rep-date').innerText = session.date;
    document.getElementById('rep-time').innerText = `${session.startTime} - ${session.endTime} น.`;
    document.getElementById('rep-late').innerText = session.lateAfter ? session.lateAfter + ' น.' : 'ไม่มีการบันทึกสาย';
    document.getElementById('rep-location').innerText = session.location || 'ไม่ระบุสถานที่';
    document.getElementById('rep-instructor').innerText = session.instructorName || 'ไม่ระบุผู้สอน';

    // Map attendance check-ins for this session
    const attendanceMap = new Map();
    appState.attendance.forEach(a => {
      if (a.sessionId === sessionId && a.studentId) {
        attendanceMap.set(a.studentId.trim().toUpperCase(), a);
      }
    });

    let ontimeCount = 0;
    let lateCount = 0;
    let absentCount = 0;

    const reportData = appState.students.map(student => {
      const studentKey = student.id.trim().toUpperCase();
      const record = attendanceMap.get(studentKey);
      
      let checkinTime = '---';
      let statusText = 'ขาดเรียน';
      let statusClass = 'text-red';
      let statusVal = 'absent';
      
      if (record) {
        statusVal = record.status;
        const timeObj = new Date(record.time);
        checkinTime = timeObj.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) + ' น.';
        
        if (record.status === 'late') {
          statusText = 'สาย';
          statusClass = 'text-orange';
          lateCount++;
        } else {
          statusText = 'ตรงเวลา';
          statusClass = 'text-green';
          ontimeCount++;
        }
      } else {
        absentCount++;
      }
      
      return {
        id: student.id,
        name: student.name,
        checkinTime,
        statusText,
        statusClass,
        statusVal
      };
    });

    window.currentSessionReport = {
      sessionId,
      session,
      report: reportData
    };

    // Render report table
    renderSessionReportTable(reportData);

    // Update session mini stats
    document.getElementById('rep-stat-ontime').innerText = ontimeCount;
    document.getElementById('rep-stat-late').innerText = lateCount;
    document.getElementById('rep-stat-absent').innerText = absentCount;

    // Calculate Overall Course Attendance Percentage for the subject code of this session
    const subjectCode = session.subjectCode;
    let overallPercentage = 0;
    if (subjectCode) {
      const courseSessions = appState.sessions.filter(s => s.subjectCode === subjectCode);
      const totalSessionsCount = courseSessions.length;
      const totalStudentsCount = appState.students.length;
      const maxPossiblePresent = totalSessionsCount * totalStudentsCount;
      
      let presentCount = 0;
      const courseSessionIds = new Set(courseSessions.map(s => s.id));
      appState.attendance.forEach(a => {
        if (courseSessionIds.has(a.sessionId)) {
          presentCount++;
        }
      });
      overallPercentage = maxPossiblePresent > 0 ? (presentCount / maxPossiblePresent) * 100 : 0;
    }
    
    const overallPctEl = document.getElementById('course-overall-pct');
    overallPctEl.innerText = `${overallPercentage.toFixed(2)}%`;
    
    const overallBarEl = document.getElementById('course-overall-progress-bar');
    overallBarEl.style.width = `${overallPercentage}%`;
    
    if (overallPercentage >= 80) {
      overallPctEl.style.color = 'var(--success)';
      overallBarEl.style.backgroundColor = 'var(--success)';
    } else if (overallPercentage >= 50) {
      overallPctEl.style.color = 'var(--warning)';
      overallBarEl.style.backgroundColor = 'var(--warning)';
    } else {
      overallPctEl.style.color = 'var(--danger)';
      overallBarEl.style.backgroundColor = 'var(--danger)';
    }

    const btnExport = document.getElementById('btn-export-course-csv');
    if (btnExport) btnExport.style.display = 'inline-block';
    panel.classList.remove('hide');
  } catch (err) {
    console.error(err);
    const btnExport = document.getElementById('btn-export-course-csv');
    if (btnExport) btnExport.style.display = 'none';
    if (tbody) tbody.innerHTML = `<tr><td colspan="4" class="text-center text-red">ดึงข้อมูลรายงานล้มเหลว: ${err.message}</td></tr>`;
  }
}

// Render session report data into HTML table
function renderSessionReportTable(reportData) {
  const tbody = document.getElementById('course-report-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (reportData.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">ไม่มีข้อมูลนักศึกษาในระบบ</td></tr>';
    return;
  }

  // Sort by Student ID
  const sorted = [...reportData].sort((a, b) => a.id.localeCompare(b.id));

  sorted.forEach(r => {
    const row = document.createElement('tr');
    row.className = 'course-report-row';
    row.setAttribute('data-id', r.id.toLowerCase());
    row.setAttribute('data-name', r.name.toLowerCase());

    row.innerHTML = `
      <td><strong>${r.id}</strong></td>
      <td>${r.name}</td>
      <td>${r.checkinTime}</td>
      <td class="text-center ${r.statusClass}" style="font-weight: 600;">${r.statusText}</td>
    `;
    tbody.appendChild(row);
  });
}

// Client-side search filtering for student session attendance table
function filterSessionReportStudents() {
  const query = document.getElementById('course-student-search').value.toLowerCase().trim();
  const rows = document.querySelectorAll('.course-report-row');
  
  rows.forEach(row => {
    const id = row.getAttribute('data-id');
    const name = row.getAttribute('data-name');
    if (id.includes(query) || name.includes(query)) {
      row.style.display = '';
    } else {
      row.style.display = 'none';
    }
  });
}

// Export session report to CSV
function exportSessionReportToCSV() {
  if (!window.currentSessionReport || !window.currentSessionReport.report) {
    alert('กรุณาเลือกคาบเรียนและโหลดข้อมูลก่อนทำการส่งออก');
    return;
  }
  
  const { session, report } = window.currentSessionReport;
  
  let csvContent = "\uFEFF"; // UTF-8 BOM
  csvContent += `รายงานการเข้าเรียนรายคาบ,,,\n`;
  csvContent += `วิชา,${session.subjectName || session.className},,\n`;
  csvContent += `รหัสวิชา,${session.subjectCode || 'N/A'},,\n`;
  csvContent += `วันที่เรียน,${session.date},,\n`;
  csvContent += `เวลาเรียน,${session.startTime} - ${session.endTime} น.,,\n`;
  csvContent += `ผู้สอน,${session.instructorName || 'ไม่ระบุ'},,\n`;
  csvContent += `\n`;
  csvContent += `รหัสนักศึกษา,ชื่อ-นามสกุล,เวลาเช็คชื่อ,สถานะ\n`;
  
  report.forEach(r => {
    const id = `"${r.id.replace(/"/g, '""')}"`;
    const name = `"${r.name.replace(/"/g, '""')}"`;
    
    csvContent += `${id},${name},${r.checkinTime},${r.statusText}\n`;
  });
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  
  const sanitizedCode = (session.subjectCode || 'Session').replace(/[^a-zA-Z0-9-]/g, '_');
  link.setAttribute("download", `Session_Report_${sanitizedCode}_${session.date}.csv`);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
// Render list of system users (Admin only)
function renderUsersTable() {
  const tbody = document.getElementById('users-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  const users = appState.users || [];
  if (users.length === 0) {
    tbody.innerHTML = '<tr><td colspan="2" class="text-center text-muted">ไม่พบข้อมูลผู้ใช้ในระบบ</td></tr>';
    return;
  }

  const currentLoggedInUser = sessionStorage.getItem('username');

  // Sort: admin first, then alphabetical
  const sorted = [...users].sort((a, b) => {
    if (a.username.toLowerCase() === 'admin') return -1;
    if (b.username.toLowerCase() === 'admin') return 1;
    return a.username.localeCompare(b.username);
  });

  sorted.forEach(u => {
    const row = document.createElement('tr');
    
    // Disable delete for default 'admin' or currently logged in user
    const cannotDelete = u.username.toLowerCase() === 'admin' || u.username === currentLoggedInUser;
    
    const deleteBtn = cannotDelete
      ? `<button class="btn btn-secondary btn-xs" disabled style="opacity: 0.5; cursor: not-allowed;" title="ไม่สามารถลบผู้ใช้หลักหรือบัญชีที่กำลังใช้งานอยู่ได้"><i class="fa-solid fa-ban"></i> ลบไม่ได้</button>`
      : `<button class="btn btn-danger-outline btn-xs" onclick="handleDeleteUser('${u.username}')"><i class="fa-solid fa-user-xmark"></i> ลบผู้ใช้</button>`;

    row.innerHTML = `
      <td><strong>${u.username}</strong></td>
      <td class="text-center">${deleteBtn}</td>
    `;
    
    tbody.appendChild(row);
  });
}

// Add User handler (Admin only)
async function handleAddUser(e) {
  e.preventDefault();
  const usernameInput = document.getElementById('new-username');
  const passwordInput = document.getElementById('new-password');
  
  const username = usernameInput.value.trim();
  const password = passwordInput.value;
  const role = 'teacher'; // Auto-assign teacher role

  if (!username || !password) {
    alert('กรุณากรอกข้อมูลให้ครบถ้วน');
    return;
  }

  // Username validation (English letters, numbers, underscores only)
  const usernameRegex = /^[a-zA-Z0-9_]+$/;
  if (!usernameRegex.test(username)) {
    alert('ชื่อผู้ใช้งานต้องเป็นภาษาอังกฤษ ตัวเลข หรือเครื่องหมาย underscore (_) เท่านั้น');
    return;
  }

  try {
    const res = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, role })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'เพิ่มบัญชีไม่สำเร็จ');
    }

    usernameInput.value = '';
    passwordInput.value = '';

    alert(`เพิ่มอาจารย์ "${username}" เรียบร้อยแล้ว!`);
    fetchDatabase();
  } catch (err) {
    alert('เกิดข้อผิดพลาด: ' + err.message);
  }
}

// Delete User handler (Admin only)
async function handleDeleteUser(username) {
  if (!confirm(`คุณแน่ใจหรือไม่ว่าต้องการลบผู้ใช้งาน "${username}" ?`)) {
    return;
  }

  try {
    const res = await fetch(`/api/users/${username}`, {
      method: 'DELETE'
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'ลบผู้ใช้ไม่สำเร็จ');
    }

    alert(`ลบผู้ใช้ "${username}" สำเร็จ`);
    fetchDatabase();
  } catch (err) {
    alert('เกิดข้อผิดพลาด: ' + err.message);
  }
}
