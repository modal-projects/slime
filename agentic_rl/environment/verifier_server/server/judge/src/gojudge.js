import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import FormData from 'form-data';
import {createReadStream} from 'fs';


export class GoJudgeClient {
    constructor(baseURL = 'http://127.0.0.1:5050') {
        this.baseURL = baseURL;
    }

    // POST /run with bounded retry on transient backpressure. go-judge has a bounded
    // work queue; when a burst of /run requests exceeds (parallelism + queue) it rejects
    // with HTTP 500 "worker queue is full" BEFORE executing — so the request never ran and
    // retrying is safe/idempotent. Without this, high worker counts (nproc 32+) let many
    // submissions' per-case fan-out flood go-judge at once and grades spuriously fail.
    async _postRun(payload) {
        const maxAttempts = 8;
        let delay = 150;
        for (let attempt = 1; ; attempt++) {
            try {
                const { data } = await axios.post(`${this.baseURL}/run`, payload, { timeout: 300000 });
                return data;
            } catch (e) {
                const body = e?.response?.data;
                const msg = typeof body === 'string' ? body : JSON.stringify(body || '');
                const retriable = (e?.response?.status >= 500 && /queue is full|too many/i.test(msg))
                    || e?.code === 'ECONNRESET' || e?.code === 'ECONNREFUSED' || e?.code === 'ECONNABORTED';
                if (!retriable || attempt >= maxAttempts) throw e;
                const wait = delay + Math.floor(Math.random() * delay);
                await new Promise(r => setTimeout(r, wait));
                delay = Math.min(delay * 2, 3000);
            }
        }
    }

    // Basic invocation
    async runOne(cmd) {
        const data = await this._postRun({ cmd: [cmd] });
        return data[0];
    }

    async run(cmds) {
        return await this._postRun(cmds);
    }

    // Delete file
    async deleteFile(fileId) {
        try { 
            await axios.delete(`${this.baseURL}/file/${encodeURIComponent(fileId)}`); 
        } catch { }
    }

    // Get file content
    async getFileContent(fileId) {
        const { data } = await axios.get(`${this.baseURL}/file/${encodeURIComponent(fileId)}`, { responseType: 'arraybuffer' });
        return data;
    }

    // Upload file
    async copyInFile(filePath) {
        try {
            const form = new FormData();
            // 'file' is the field name required by backend, fs.createReadStream for efficient file reading
            form.append('file', createReadStream(filePath));
            // Use await to wait for post request completion
            const { data } = await axios.post(`${this.baseURL}/file`, form, {
                headers: {
                    ...form.getHeaders() // form-data library generates necessary multipart headers
                }
            });
            return data;
        } catch (error) {
            console.error(`Failed to upload file ${filePath}:`, error.message);
            throw error; // Throw exception up for caller to handle
        }
    }


    // Cache single file
    async cacheSingleFile(name, content) {
        const res = await this.runOne({
            args: ['/bin/true'],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 1 }, { name: 'stderr', max: 1024 }],
            copyIn: { [name]: { content } },
            copyOutCached: [name],
            cpuLimit: 1e9, 
            memoryLimit: 16 << 20, 
            procLimit: 5,
        });
        if (res.status !== 'Accepted' || !res.fileIds?.[name]) {
            throw new Error(`cache file failed: ${res.status}`);
        }
        return res.fileIds[name];
    }

    // Prepare program (compile or cache)
    async prepareProgram({ lang, code, mainName = null, testlibPath = '/lib/testlib' }) {
        if (lang === 'cpp') {
            return await this._prepareCpp(code, mainName);
        }
        if (lang === 'java') {
            return await this._prepareJava(code, mainName);
        }
        if (lang === 'py' || lang === 'pypy' || lang === 'python' || lang === 'python3') {
            return await this._preparePython(code, mainName, lang);
        }
        throw new Error('unsupported lang');
    }

    async _prepareCpp(code, mainName) {
        const srcName = mainName || 'main.cpp';
        const outName = 'a';
        const res = await this.runOne({
            args: ['/usr/bin/g++', srcName, '-O2', '-pipe', '-std=gnu++17', '-o', outName],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 1024 }, { name: 'stderr', max: 1024 * 1024 }],
            copyIn: { [srcName]: { content: code } },
            copyOut: ['stdout', 'stderr'],
            copyOutCached: [outName],
            cpuLimit: 10e9, 
            memoryLimit: 512 << 20, 
            procLimit: 50,
        });
        if (res.status !== 'Accepted') {
            throw new Error(`compile failed: ${res.files?.stderr || res.status}`);
        }
        const exeId = res.fileIds[outName];
        return { 
            runArgs: [outName], 
            preparedCopyIn: { [outName]: { fileId: exeId } }, 
            cleanupIds: [exeId] 
        };
    }

    async _prepareJava(code, mainName) {
        const srcName = mainName || 'Main.java';
        const mainClass = (srcName.replace(/\.java$/, '') || 'Main');
        const res = await this.runOne({
            args: ['/usr/bin/javac', srcName],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 64 }, { name: 'stderr', max: 1024 * 64 }],
            copyIn: { [srcName]: { content: code } },
            copyOut: ['stdout', 'stderr'],
            copyOutCached: [`${mainClass}.class`],
            cpuLimit: 10e9, 
            memoryLimit: 1024 << 20, 
            procLimit: 50,
        });
        if (res.status !== 'Accepted') {
            throw new Error(`javac failed: ${res.files?.stderr || res.status}`);
        }
        const clsId = res.fileIds[`${mainClass}.class`];
        return {
            runArgs: ['/usr/bin/java', mainClass],
            preparedCopyIn: { [`${mainClass}.class`]: { fileId: clsId } },
            cleanupIds: [clsId],
        };
    }

    async _preparePython(code, mainName, lang) {
        const srcName = mainName || 'main.py';
        const fileId = await this.cacheSingleFile(srcName, code);
        const interp = (lang === 'pypy') ? '/usr/bin/pypy3' : '/usr/bin/python3';
        return { 
            runArgs: [interp, srcName], 
            preparedCopyIn: { [srcName]: { fileId } }, 
            cleanupIds: [fileId] 
        };
    }

    // Prepare checker
    async prepareChecker(checkerSourceText, testlibPath = '/lib/testlib', srcName = 'chk.cc') {
        const outName = 'chk';
        const res = await this.runOne({
            args: ['/usr/bin/g++', srcName, '-O2', '-pipe', '-std=gnu++17', '-I', testlibPath, '-o', outName],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 64 }, { name: 'stderr', max: 1024 * 64 }],
            copyIn: { [srcName]: { content: checkerSourceText } },
            copyOutCached: [outName],
            cpuLimit: 30e9,  // 30s (increased from 10s to handle high load)
            memoryLimit: 512 << 20,
            procLimit: 50,
        });
        if (res.status !== 'Accepted') {
            throw new Error(`checker compile failed: ${res.files?.stderr || res.status}`);
        }
        const checkerId = res.fileIds[outName];
        return {
            checkerId,
            cleanup: () => this.deleteFile(checkerId)
        };
    }

    async prepareInteractor(interactorSourceText, testlibPath = '/lib/testlib', srcName = 'interactor.cc') {
        const outName = 'interactor';
        const res = await this.runOne({
            args: ['/usr/bin/g++', srcName, '-O2', '-pipe', '-std=gnu++17', '-I', testlibPath, '-o', outName],
            env: ['PATH=/usr/bin:/bin'],
            files: [{ content: '' }, { name: 'stdout', max: 1024 * 1024 }, { name: 'stderr', max: 1024 * 1024 }],
            copyIn: { [srcName]: { content: interactorSourceText } },
            copyOutCached: [outName],
            cpuLimit: 30e9,  // 30s (increased from 10s to handle high load)
            memoryLimit: 512 << 20,
            procLimit: 128,
        });
        if (res.status !== 'Accepted') {
            throw new Error(`interactor compile failed: ${res.files?.stderr || res.status}`);
        }
        const interactorId = res.fileIds[outName];
        return {
            interactorId,
            cleanup: () => this.deleteFile(interactorId)
        };
    }

    async getCheckerBin(checkerSourceText, testlibPath = '/lib/testlib', srcName = 'chk.cc') {
        const { checkerId, cleanup } = await this.prepareChecker(checkerSourceText, testlibPath, srcName);
        const checkerBin = await this.getFileContent(checkerId);
        cleanup();
        return checkerBin;
    }

    async getInteractorBin(interactorSourceText, testlibPath = '/lib/testlib', srcName = 'interactor.cc') {
        const { interactorId, cleanup } = await this.prepareInteractor(interactorSourceText, testlibPath, srcName);
        const interactorBin = await this.getFileContent(interactorId);
        cleanup();
        return interactorBin;
    }

    async copyInBin(binPath, testlibPath = '/lib/testlib', srcName = 'chk.cc') {
        if (!binPath) {
            throw new Error('binPath is required');
        }
        const binId = await this.copyInFile(binPath);
        if (!binId) {
            throw new Error('Failed to copy in binary');
        }
        const cleanup = () => this.deleteFile(binId);
        return { binId, cleanup };
    }
}