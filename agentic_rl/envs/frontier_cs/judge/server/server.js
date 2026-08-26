import express from 'express';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

import { SubmissionManager } from './src/utils.js';
import { ProblemManager } from './src/problem_manager.js';
import { JudgeEngine } from './src/judge_engine.js';
import { createApiRoutes } from './src/router.js';

// Safety nets: on Node 20 an unhandled rejection (e.g. a grading job throwing
// outside its try/catch under burst load) terminates the process, taking every
// in-flight submission with it. Log and keep serving instead.
process.on('unhandledRejection', (reason) => {
    console.error('[server] unhandledRejection:', reason);
});
process.on('uncaughtException', (err) => {
    console.error('[server] uncaughtException:', err);
});

// Initialize configuration
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const config = {
    problemsRoot: process.env.PROBLEMS_ROOT || path.join(__dirname, 'problems'),
    dataRoot: path.join(__dirname, 'data'),
    submissionsRoot: process.env.SUBMISSIONS_DIR || path.join(__dirname, 'submissions'),
    bucketSize: +(process.env.SUB_BUCKET || 100),
    gjAddr: process.env.GJ_ADDR || 'http://127.0.0.1:5050',
    workers: +(process.env.JUDGE_WORKERS || 4),
    testlibPath: process.env.TESTLIB_INSIDE || '/lib/testlib',
    port: process.env.PORT || 8081
};

// Create directories
await fs.mkdir(config.dataRoot, { recursive: true });
await fs.mkdir(config.submissionsRoot, { recursive: true });

// Initialize modules
const submissionManager = new SubmissionManager(
    config.dataRoot, 
    config.submissionsRoot, 
    config.bucketSize
);

const problemManager = new ProblemManager({
    problemsRoot: config.problemsRoot,
    gjAddr: config.gjAddr,
    testlibPath: config.testlibPath
});

const judgeEngine = new JudgeEngine({
    problemsRoot: config.problemsRoot,
    gjAddr: config.gjAddr,
    submissionManager,
    testlibPath: config.testlibPath,
    workers: config.workers
});

// Create Express application
const app = express();
app.use(express.json({ limit: '10mb' }));

// Idle self-termination: the only traffic this server ever gets is from the
// training run that booted it (grading requests + /health probes at least
// every ~60s while episodes flow). If the owner dies — app stop, crash,
// preemption — traffic stops entirely, so after JUDGE_IDLE_EXIT_MIN of
// silence we exit with IDLE_EXIT_CODE, which the entrypoint treats as
// "terminate the sandbox" rather than a crash to restart. A false positive
// is harmless: autostart's health recheck reboots a replacement judge.
const IDLE_EXIT_CODE = 86;
const idleExitMin = +(process.env.JUDGE_IDLE_EXIT_MIN ?? 25);
let lastRequestAt = Date.now();
app.use((req, res, next) => {
    lastRequestAt = Date.now();
    next();
});
if (idleExitMin > 0) {
    setInterval(() => {
        const idleMin = (Date.now() - lastRequestAt) / 60000;
        if (idleMin >= idleExitMin) {
            console.error(
                `[server] no requests for ${idleMin.toFixed(1)} min (limit ${idleExitMin}); ` +
                `owner presumed dead — exiting ${IDLE_EXIT_CODE} to terminate the sandbox`
            );
            process.exit(IDLE_EXIT_CODE);
        }
    }, 60000).unref();
}

// Register routes
const apiRoutes = createApiRoutes(judgeEngine, problemManager, submissionManager);
app.use('/', apiRoutes);

// Start server
app.listen(config.port, () => {
    console.log(`LightCPVerifier listening on port ${config.port} (modular architecture)`);
});

export { judgeEngine, problemManager, submissionManager };