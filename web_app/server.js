const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const http = require('http');
const os = require('os');

function getLocalIpAddress() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return 'localhost';
}

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve photos from the faces folder as static assets
app.use('/faces', express.static(path.join(__dirname, '../faces')));

const DB_FILE = path.join(__dirname, 'db.json');
const FACES_DIR = path.join(__dirname, '../faces');

// Ensure database and faces folder exist
if (!fs.existsSync(FACES_DIR)) {
  fs.mkdirSync(FACES_DIR, { recursive: true });
}

// Database initial state
let db = {
  students: [],
  sessions: [],
  attendance: [],
  users: []
};

// Load database
function loadDb() {
  if (fs.existsSync(DB_FILE)) {
    try {
      db = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    } catch (e) {
      console.error('Error reading database file:', e);
    }
  } else {
    saveDb();
  }
  // Initialize default admin user if none exists
  if (!db.users || db.users.length === 0) {
    db.users = [
      { username: 'admin', password: '123' }
    ];
    saveDb();
  }
  // Sync directories to database to register any folders already in faces/
  syncFacesFolderToDb();
}

// Save database
function saveDb() {
  try {
    fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2), 'utf8');
  } catch (e) {
    console.error('Error saving database:', e);
  }
}

// Sync local faces folder with the db
function syncFacesFolderToDb() {
  if (!fs.existsSync(FACES_DIR)) return;

  const dirs = fs.readdirSync(FACES_DIR);
  let changed = false;
  const newStudents = [];
  
  // Big O Optimization: Pre-collect existing IDs into a Set for O(1) lookups
  // This reduces complexity from O(N * M) to O(N + M)
  const existingIds = new Set(db.students.map(s => s.id));

  dirs.forEach(dirName => {
    const fullPath = path.join(FACES_DIR, dirName);
    if (fs.statSync(fullPath).isDirectory() && dirName.includes('-')) {
      const parts = dirName.split('-');
      const id = parts[0].trim();
      const name = parts[1].trim();

      if (!existingIds.has(id)) {
        const studentObj = {
          id: id,
          name: name,
          registeredAt: new Date().toISOString()
        };
        db.students.push(studentObj);
        newStudents.push(studentObj);
        existingIds.add(id); // Avoid adding duplicates if folder has multiple matching directories
        changed = true;
      }
    }
  });

  if (changed) {
    saveDb();
    newStudents.forEach(std => {
      sendSseAlert('student_registered', { student: std });
    });
  }
}

function autoDeactivateExpiredSessions() {
  const now = new Date();
  const offset = now.getTimezoneOffset();
  const localNow = new Date(now.getTime() - (offset * 60 * 1000));
  const todayStr = localNow.toISOString().split('T')[0];
  const currentTimeStr = localNow.toISOString().split('T')[1].substring(0, 5);

  let changed = false;
  db.sessions.forEach(s => {
    if (s.active) {
      if (s.date < todayStr || (s.date === todayStr && s.endTime < currentTimeStr)) {
        s.active = false;
        changed = true;
      }
    }
  });

  if (changed) {
    saveDb();
    sendSseAlert('session_change', { activeSession: null });
  }
}

loadDb();

// Multer Storage Configuration for student photo registration
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const studentId = req.body.studentId.trim();
    const name = req.body.name.trim();
    const folderName = `${studentId}-${name}`;
    const targetDir = path.join(FACES_DIR, folderName);

    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }
    cb(null, targetDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname) || '.jpg';
    cb(null, `photo_${uniqueSuffix}${ext}`);
  }
});

const upload = multer({ storage: storage });

// Real-time notification clients (SSE)
let sseClients = [];

function sendSseAlert(type, payload) {
  const data = JSON.stringify({ type, payload });
  sseClients.forEach(client => {
    client.write(`data: ${data}\n\n`);
  });
}

// Trigger python reload
function notifyPythonReload() {
  const options = {
    hostname: 'localhost',
    port: 5001,
    path: '/reload',
    method: 'POST',
    timeout: 2000
  };

  const req = http.request(options, (res) => {
    console.log(`Python server responded: ${res.statusCode}`);
  });

  req.on('error', (e) => {
    console.log('Could not notify Python reload (Python GUI app might be offline):', e.message);
  });

  req.end();
}

// SSE Connection Endpoint
app.get('/api/alerts', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  res.write('\n');
  sseClients.push(res);

  req.on('close', () => {
    sseClients = sseClients.filter(client => client !== res);
  });
});

// REST API Endpoints

// Get current system data
app.get('/api/db', (req, res) => {
  autoDeactivateExpiredSessions();
  syncFacesFolderToDb();
  res.json({
    ...db,
    serverIp: getLocalIpAddress()
  });
});

// Login endpoint
app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ error: 'กรุณากรอกชื่อผู้ใช้และรหัสผ่าน' });
  }

  const user = db.users.find(u => u.username === username.trim() && u.password === password);
  if (!user) {
    return res.status(401).json({ error: 'ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง' });
  }

  res.json({ success: true, token: 'session_active_' + user.username, username: user.username });
});

// Sync students manually from faces/ folder
app.post('/api/students/sync', (req, res) => {
  try {
    syncFacesFolderToDb();
    res.json({ success: true, students: db.students });
  } catch (e) {
    res.status(500).json({ error: 'การซิงค์รายชื่อล้มเหลว: ' + e.message });
  }
});

// Get session report with ontime, late, and absent statuses
app.get('/api/sessions/:id/report', (req, res) => {
  const { id } = req.params;
  const session = db.sessions.find(s => s.id === id);
  if (!session) {
    return res.status(404).json({ error: 'ไม่พบคาบเรียนนี้' });
  }

  // Big O Optimization: Map sessionId attendance records to studentId -> record
  // This reduces complexity from O(Students * Attendance) to O(Students + Attendance)
  const attendanceMap = new Map();
  db.attendance.forEach(a => {
    if (a.sessionId === id) {
      attendanceMap.set(a.studentId, a);
    }
  });

  const report = db.students.map(student => {
    const record = attendanceMap.get(student.id);
    if (record) {
      return {
        studentId: student.id,
        studentName: student.name,
        time: record.time,
        status: record.status || 'ontime',
        offline: record.offline || false
      };
    } else {
      return {
        studentId: student.id,
        studentName: student.name,
        time: null,
        status: 'absent',
        offline: false
      };
    }
  });

  res.json({ session, report });
});

// Create attendance session
app.post('/api/sessions', (req, res) => {
  const { subjectCode, subjectName, date, startTime, endTime, lateAfter, location, instructorName } = req.body;
  if (!subjectCode || !subjectName || !date || !startTime || !endTime) {
    return res.status(400).json({ error: 'กรุณากรอกรหัสวิชา, ชื่อวิชา, วันที่, เวลาเริ่ม และเวลาจบ' });
  }

  // Deactivate other sessions
  db.sessions.forEach(s => s.active = false);

  const newSession = {
    id: 'session_' + Date.now(),
    subjectCode: subjectCode.trim(),
    subjectName: subjectName.trim(),
    className: `${subjectCode.trim()} - ${subjectName.trim()}`,
    date,
    startTime,
    endTime,
    lateAfter: lateAfter || startTime,
    location: location ? location.trim() : '',
    instructorName: instructorName ? instructorName.trim() : '',
    active: true,
    createdAt: new Date().toISOString()
  };

  db.sessions.push(newSession);
  saveDb();

  sendSseAlert('session_change', { activeSession: newSession });
  res.json({ success: true, session: newSession });
});

// Toggle session active status
app.post('/api/sessions/toggle', (req, res) => {
  const { sessionId } = req.body;
  const session = db.sessions.find(s => s.id === sessionId);
  if (!session) {
    return res.status(404).json({ error: 'Session not found' });
  }

  const newState = !session.active;
  if (newState) {
    // If turning on, turn off all other sessions
    db.sessions.forEach(s => s.active = false);
  }
  session.active = newState;
  saveDb();

  sendSseAlert('session_change', { activeSession: newState ? session : null });
  res.json({ success: true, session });
});

// Delete a session
app.delete('/api/sessions/:id', (req, res) => {
  const { id } = req.params;
  db.sessions = db.sessions.filter(s => s.id !== id);
  db.attendance = db.attendance.filter(a => a.sessionId !== id);
  saveDb();
  res.json({ success: true });
});

// Register new student (upload multiple photos + metadata)
app.post('/api/students', upload.array('photo', 15), (req, res) => {
  const { studentId, name } = req.body;
  if (!studentId || !name) {
    return res.status(400).json({ error: 'Student ID and Name are required' });
  }

  // Clean IDs and names
  const cleanId = studentId.trim();
  const cleanName = name.trim();

  // Check if student already exists
  let student = db.students.find(s => s.id === cleanId);
  if (!student) {
    student = {
      id: cleanId,
      name: cleanName,
      registeredAt: new Date().toISOString()
    };
    db.students.push(student);
  } else {
    // Update name if changed
    student.name = cleanName;
  }

  saveDb();
  
  // Trigger Python reload
  notifyPythonReload();

  sendSseAlert('student_registered', { student });
  res.json({ success: true, student });
});

// Edit student (Update ID and/or Name)
app.put('/api/students/:id', (req, res) => {
  const { id } = req.params;
  const { newId, newName } = req.body;

  if (!newId || !newName) {
    return res.status(400).json({ error: 'กรุณากรอกรหัสนักศึกษาและชื่อใหม่' });
  }

  const cleanNewId = newId.trim();
  const cleanNewName = newName.trim();

  // Find student in db
  const student = db.students.find(s => s.id === id);
  if (!student) {
    return res.status(404).json({ error: 'ไม่พบข้อมูลนักศึกษานี้ในระบบ' });
  }

  // Check if new ID already exists for another student
  if (cleanNewId !== id) {
    const existing = db.students.find(s => s.id === cleanNewId);
    if (existing) {
      return res.status(400).json({ error: 'รหัสนักศึกษาใหม่นี้มีอยู่ในระบบแล้ว' });
    }
  }

  // Rename folder in faces/
  try {
    if (fs.existsSync(FACES_DIR)) {
      const dirs = fs.readdirSync(FACES_DIR);
      // Find old folder starting with old "id-"
      const oldFolderName = dirs.find(d => d.startsWith(id + '-'));
      if (oldFolderName) {
        const oldFullPath = path.join(FACES_DIR, oldFolderName);
        const newFolderName = `${cleanNewId}-${cleanNewName}`;
        const newFullPath = path.join(FACES_DIR, newFolderName);
        if (oldFullPath !== newFullPath) {
          fs.renameSync(oldFullPath, newFullPath);
        }
      }
    }
  } catch (err) {
    console.error('Error renaming student folder:', err);
    return res.status(500).json({ error: 'ไม่สามารถเปลี่ยนชื่อโฟลเดอร์รูปภาพได้: ' + err.message });
  }

  // Update student properties
  student.id = cleanNewId;
  student.name = cleanNewName;

  // Update in attendance records if any
  db.attendance.forEach(a => {
    if (a.studentId === id) {
      a.studentId = cleanNewId;
      a.studentName = cleanNewName;
    }
  });

  saveDb();
  notifyPythonReload();
  sendSseAlert('student_updated', { student, oldId: id });
  res.json({ success: true, student });
});

// Delete student
app.delete('/api/students/:id', (req, res) => {
  const { id } = req.params;

  const studentIndex = db.students.findIndex(s => s.id === id);
  if (studentIndex === -1) {
    return res.status(404).json({ error: 'ไม่พบนักศึกษาในระบบ' });
  }

  const student = db.students[studentIndex];

  // Delete folder in faces/
  try {
    if (fs.existsSync(FACES_DIR)) {
      const dirs = fs.readdirSync(FACES_DIR);
      const studentFolderName = dirs.find(d => d.startsWith(id + '-'));
      if (studentFolderName) {
        const fullPath = path.join(FACES_DIR, studentFolderName);
        fs.rmSync(fullPath, { recursive: true, force: true });
      }
    }
  } catch (err) {
    console.error('Error deleting student folder:', err);
    return res.status(500).json({ error: 'ไม่สามารถลบโฟลเดอร์รูปภาพได้: ' + err.message });
  }

  // Remove from students array
  db.students.splice(studentIndex, 1);

  // Remove attendance records of this student
  db.attendance = db.attendance.filter(a => a.studentId !== id);

  saveDb();
  notifyPythonReload();
  sendSseAlert('student_deleted', { studentId: id });
  res.json({ success: true });
});

// Attendance Check-in API (called by python script)
app.post('/api/attendance/checkin', (req, res) => {
  autoDeactivateExpiredSessions();
  const { name_id } = req.body;
  if (!name_id || name_id === 'Unknown') {
    sendSseAlert('checkin_unknown', { time: new Date().toISOString() });
    return res.status(400).json({ status: 'unknown', message: 'ไม่พบข้อมูลใบหน้าในระบบ' });
  }

  // Parse ID and Name: B6615406-thamonawan Kroenkratok
  const parts = name_id.split('-');
  const studentId = parts[0].trim();
  const studentName = parts.length > 1 ? parts[1].trim() : '';

  // 1. Get active session
  const activeSession = db.sessions.find(s => s.active);
  if (!activeSession) {
    sendSseAlert('checkin_rejected', {
      studentId,
      studentName,
      reason: 'ไม่อยู่ในเวลาเช็คชื่อ (ไม่มีช่วงเวลาเช็คชื่อที่เปิดอยู่)',
      time: new Date().toISOString()
    });
    return res.json({ status: 'no_session', message: 'ไม่อยู่ในเวลาเช็คชื่อ (ไม่มีช่วงเวลาที่เปิดอยู่)' });
  }

  // 2. Validate session time window
  const now = new Date();
  const offset = now.getTimezoneOffset();
  const localNow = new Date(now.getTime() - (offset * 60 * 1000));
  const todayStr = localNow.toISOString().split('T')[0]; // YYYY-MM-DD
  const currentTimeStr = localNow.toISOString().split('T')[1].substring(0, 5); // HH:MM

  // Check date match
  if (activeSession.date !== todayStr) {
    sendSseAlert('checkin_rejected', {
      studentId,
      studentName,
      reason: `ไม่อยู่ในเวลาเช็คชื่อ (เซสชันกำหนดวันที่ ${activeSession.date} แต่วันนี้คือ ${todayStr})`,
      time: now.toISOString()
    });
    return res.json({ status: 'no_session', message: `ไม่อยู่ในเวลาเช็คชื่อ (เซสชันกำหนดวันที่ ${activeSession.date})` });
  }

  // Check time boundary
  if (currentTimeStr < activeSession.startTime || currentTimeStr > activeSession.endTime) {
    sendSseAlert('checkin_rejected', {
      studentId,
      studentName,
      reason: `ไม่อยู่ในเวลาเช็คชื่อ (กำหนดเวลา ${activeSession.startTime} - ${activeSession.endTime})`,
      time: now.toISOString()
    });
    return res.json({ status: 'no_session', message: `ไม่อยู่ในเวลาเช็คชื่อ (กำหนดเวลา ${activeSession.startTime} - ${activeSession.endTime})` });
  }

  // 3. Check if student already checked in for this session
  const alreadyCheckedIn = db.attendance.find(
    a => a.studentId === studentId && a.sessionId === activeSession.id
  );

  if (alreadyCheckedIn) {
    sendSseAlert('checkin_duplicate', {
      studentId,
      studentName,
      time: now.toISOString(),
      className: activeSession.className
    });
    return res.json({ 
      status: 'already_checked_in', 
      student: { id: studentId, name: studentName }, 
      message: 'เช็คชื่อไปแล้ว' 
    });
  }

  // Calculate status: ontime or late
  const checkinStatus = (activeSession.lateAfter && currentTimeStr > activeSession.lateAfter) ? 'late' : 'ontime';

  // 4. Record new check-in
  const checkinRecord = {
    studentId,
    studentName,
    sessionId: activeSession.id,
    className: activeSession.className,
    subjectCode: activeSession.subjectCode || '',
    time: now.toISOString(),
    status: checkinStatus,
    offline: false
  };

  db.attendance.push(checkinRecord);
  saveDb();

  sendSseAlert('checkin_success', checkinRecord);

  res.json({
    status: 'success',
    attendanceStatus: checkinStatus,
    student: { id: studentId, name: studentName },
    checkinTime: now.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    message: `เช็คชื่อสำเร็จ (${checkinStatus === 'ontime' ? 'ตรงเวลา' : 'สาย'})`
  });
});

// Offline Attendance Sync API (called by python script when connection restores)
app.post('/api/attendance/sync_offline', (req, res) => {
  const { records } = req.body;
  if (!Array.isArray(records) || records.length === 0) {
    return res.json({ status: 'success', synced: 0 });
  }

  // Find the active session, or the most recent session
  const activeSession = db.sessions.find(s => s.active) || db.sessions[db.sessions.length - 1];
  
  if (!activeSession) {
    return res.status(400).json({ status: 'error', message: 'No session available to sync attendance records to' });
  }

  let syncCount = 0;
  records.forEach(rec => {
    const parts = rec.name_id.split('-');
    const studentId = parts[0].trim();
    const studentName = parts.length > 1 ? parts[1].trim() : '';
    const recordTime = rec.time;

    // Check duplicate checkin in database
    const alreadyChecked = db.attendance.find(
      a => a.studentId === studentId && a.sessionId === activeSession.id
    );

    if (!alreadyChecked) {
      const recTime = new Date(recordTime);
      const recTimeStr = recTime.toTimeString().split(' ')[0].substring(0, 5); // HH:MM
      const checkinStatus = (activeSession.lateAfter && recTimeStr > activeSession.lateAfter) ? 'late' : 'ontime';

      const checkinRecord = {
        studentId,
        studentName,
        sessionId: activeSession.id,
        className: activeSession.className,
        subjectCode: activeSession.subjectCode || '',
        time: recordTime,
        status: checkinStatus,
        offline: true
      };
      db.attendance.push(checkinRecord);
      syncCount++;

      // Trigger SSE alert for each synced record
      sendSseAlert('checkin_success', checkinRecord);
    }
  });

  if (syncCount > 0) {
    saveDb();
  }

  res.json({ status: 'success', synced: syncCount });
});

// Export Database as JSON Backup
app.get('/api/backup/export', (req, res) => {
  res.setHeader('Content-disposition', 'attachment; filename=attendance_backup_' + Date.now() + '.json');
  res.setHeader('Content-type', 'application/json');
  res.write(JSON.stringify(db, null, 2));
  res.end();
});

// Start Server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
