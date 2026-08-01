/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  Clock, Play, Pause, Trash2, RefreshCw, Plus, CheckCircle2, 
  XCircle, AlertCircle, FileCode, FileText, Terminal, Server, X, Activity, Zap
} from 'lucide-react';

export interface ScheduledJob {
  id: string;
  name: string;
  source: 'workflow_component' | 'multi_agent' | 'user_direct';
  schedulerServer: 'temporal' | 'trigger_dev' | 'embedded';
  scheduleType: 'cron' | 'interval' | 'one_shot';
  cronExpression?: string;
  intervalSeconds?: number;
  actionType: 'auto_detect' | 'code_execution' | 'file_action' | 'pipeline_workflow' | 'agent_task';
  payload: string;
  detectedCategory?: 'code' | 'executable_file' | 'document_file' | 'workflow' | 'agent_task';
  targetLanguageOrExt?: string;
  status: 'scheduled' | 'running' | 'completed' | 'failed' | 'paused' | 'cancelled';
  createdAt: string;
  lastRunAt?: string;
  nextRunAt?: string;
  runCount: number;
  lastRunResult?: string;
  logs: string[];
}

interface SchedulerStatusPanelProps {
  onClose?: () => void;
}

export function SchedulerStatusPanel({ onClose }: SchedulerStatusPanelProps) {
  const [jobs, setJobs] = useState<ScheduledJob[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<ScheduledJob | null>(null);
  
  // Registration Form State
  const [showAddModal, setShowAddModal] = useState(false);
  const [newJobName, setNewJobName] = useState('');
  const [newServer, setNewServer] = useState<'temporal' | 'trigger_dev' | 'embedded'>('temporal');
  const [newScheduleType, setNewScheduleType] = useState<'cron' | 'interval'>('cron');
  const [newCronExpr, setNewCronExpr] = useState('*/5 * * * *');
  const [newIntervalSec, setNewIntervalSec] = useState(300);
  const [newPayload, setNewPayload] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const fetchJobs = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/scheduler/jobs');
      const data = await res.json();
      if (data.success) {
        setJobs(data.jobs || []);
        if (selectedJob) {
          const updated = (data.jobs || []).find((j: ScheduledJob) => j.id === selectedJob.id);
          if (updated) setSelectedJob(updated);
        }
      } else {
        setError(data.error || 'Failed to fetch scheduled jobs');
      }
    } catch (err: any) {
      setError(err.message || 'Error connecting to scheduler API');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const timer = setInterval(fetchJobs, 5000); // Poll status every 5 seconds
    return () => clearInterval(timer);
  }, []);

  const handleRegisterJob = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPayload.trim()) return;
    setIsSubmitting(true);
    try {
      const res = await fetch('/api/scheduler/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newJobName.trim() || 'Custom Schedule Job',
          source: 'user_direct',
          schedulerServer: newServer,
          scheduleType: newScheduleType,
          cronExpression: newCronExpr,
          intervalSeconds: newIntervalSec,
          actionType: 'auto_detect',
          payload: newPayload.trim()
        })
      });
      const data = await res.json();
      if (data.success) {
        setShowAddModal(false);
        setNewJobName('');
        setNewPayload('');
        await fetchJobs();
      } else {
        alert(data.error || 'Failed to register job');
      }
    } catch (err: any) {
      alert(err.message || 'Error registering job');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTriggerNow = async (id: string) => {
    try {
      const res = await fetch(`/api/scheduler/jobs/${id}/trigger`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        await fetchJobs();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleTogglePause = async (id: string) => {
    try {
      await fetch(`/api/scheduler/jobs/${id}/toggle`, { method: 'POST' });
      await fetchJobs();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteJob = async (id: string) => {
    if (!confirm('Are you sure you want to remove this schedule?')) return;
    try {
      await fetch(`/api/scheduler/jobs/${id}`, { method: 'DELETE' });
      if (selectedJob?.id === id) setSelectedJob(null);
      await fetchJobs();
    } catch (err) {
      console.error(err);
    }
  };

  const getCategoryBadge = (cat?: string, ext?: string) => {
    if (cat === 'executable_file') {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1">
          <Zap className="w-3 h-3 text-amber-400" /> Executable File (.{ext})
        </span>
      );
    }
    if (cat === 'document_file') {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-sky-500/20 text-sky-300 border border-sky-500/30 flex items-center gap-1">
          <FileText className="w-3 h-3 text-sky-400" /> Open Document (.{ext})
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
        <FileCode className="w-3 h-3 text-emerald-400" /> Execute Code ({ext || 'code'})
      </span>
    );
  };

  const getServerBadge = (server: string) => {
    if (server === 'temporal') {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-violet-500/20 text-violet-300 border border-violet-500/30">
          Temporal.io Server
        </span>
      );
    }
    if (server === 'trigger_dev') {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-pink-500/20 text-pink-300 border border-pink-500/30">
          Trigger.dev Engine
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
        Embedded Scheduler
      </span>
    );
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/95 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden font-sans">
      {/* Header */}
      <div className="px-5 py-3.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-violet-500/10 text-violet-400 border border-violet-500/20">
            <Clock className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              Temporal.io & Trigger.dev Scheduler Monitor
            </h2>
            <p className="text-xs text-slate-400">
              Manage registered cron/interval workflows, code executions & file launchers
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={fetchJobs}
            disabled={isLoading}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-mono flex items-center gap-1 cursor-pointer transition"
            title="Refresh Jobs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-indigo-400' : ''}`} />
          </button>

          <button
            type="button"
            onClick={() => setShowAddModal(true)}
            className="px-3 py-1.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 cursor-pointer shadow-md transition"
          >
            <Plus className="w-3.5 h-3.5" /> Register Schedule
          </button>

          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-12 overflow-hidden">
        {/* Left: Job List */}
        <div className="md:col-span-7 border-r border-slate-800/80 overflow-y-auto p-4 space-y-3 bg-slate-900/50">
          {error && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-xl text-xs font-mono flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {jobs.length === 0 ? (
            <div className="text-center py-12 px-4 space-y-3">
              <Server className="w-8 h-8 text-slate-600 mx-auto" />
              <div className="text-xs text-slate-400 font-mono">No active schedules registered yet.</div>
              <p className="text-[11px] text-slate-500 max-w-sm mx-auto">
                Add a schedule from a Pipeline Workflow node, multi-agent system, or click "Register Schedule" above to schedule python code, executables (.exe), or open documents (.docx).
              </p>
            </div>
          ) : (
            jobs.map((job) => {
              const isSelected = selectedJob?.id === job.id;
              return (
                <div
                  key={job.id}
                  onClick={() => setSelectedJob(job)}
                  className={`p-3.5 rounded-xl border transition-all cursor-pointer space-y-2.5 ${
                    isSelected
                      ? 'bg-violet-950/30 border-violet-500/60 shadow-lg'
                      : 'bg-slate-950/60 hover:bg-slate-850/80 border-slate-800'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold font-mono text-slate-200 truncate">
                          {job.name}
                        </span>
                        {job.status === 'scheduled' && (
                          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" title="Scheduled / Active" />
                        )}
                        {job.status === 'paused' && (
                          <span className="w-2 h-2 rounded-full bg-amber-400" title="Paused" />
                        )}
                      </div>
                      <div className="text-[10px] font-mono text-slate-500 truncate mt-0.5">
                        ID: {job.id} • Source: <span className="text-slate-400">{job.source}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-1 shrink-0">
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleTriggerNow(job.id); }}
                        className="p-1 bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-300 rounded border border-emerald-500/30 text-[10px] font-mono flex items-center gap-1 cursor-pointer"
                        title="Trigger Execution Now"
                      >
                        <Play className="w-3 h-3" /> Run Now
                      </button>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleTogglePause(job.id); }}
                        className="p-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px] cursor-pointer"
                        title={job.status === 'paused' ? 'Resume Schedule' : 'Pause Schedule'}
                      >
                        {job.status === 'paused' ? <Play className="w-3 h-3 text-amber-400" /> : <Pause className="w-3 h-3 text-slate-400" />}
                      </button>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleDeleteJob(job.id); }}
                        className="p-1 bg-rose-950/30 hover:bg-rose-900/50 text-rose-400 rounded text-[10px] cursor-pointer"
                        title="Delete Schedule"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5 pt-1 border-t border-slate-800/60">
                    {getServerBadge(job.schedulerServer)}
                    {getCategoryBadge(job.detectedCategory, job.targetLanguageOrExt)}
                    <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                      Cron: {job.scheduleType === 'cron' ? job.cronExpression : `${job.intervalSeconds}s`}
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 pt-0.5">
                    <span>Runs: <strong className="text-slate-300">{job.runCount}</strong></span>
                    <span>Next: <strong className="text-violet-300">{job.nextRunAt ? new Date(job.nextRunAt).toLocaleTimeString() : 'N/A'}</strong></span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Right: Selected Job Details & Execution History */}
        <div className="md:col-span-5 bg-slate-950/80 p-4 flex flex-col h-full overflow-y-auto space-y-4">
          {selectedJob ? (
            <div className="space-y-4">
              <div className="border-b border-slate-800 pb-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold font-mono text-slate-100">{selectedJob.name}</h3>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold bg-violet-950 text-violet-300 border border-violet-500/30">
                    {selectedJob.status}
                  </span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-1">
                  Registered: {new Date(selectedJob.createdAt).toLocaleString()}
                </div>
              </div>

              {/* Payload Inspection */}
              <div className="space-y-1.5">
                <div className="text-[11px] font-mono font-bold text-slate-300 uppercase tracking-wider flex items-center justify-between">
                  <span>Target Payload / Path</span>
                  <span className="text-[10px] text-violet-400">{selectedJob.detectedCategory}</span>
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 font-mono text-xs text-indigo-200 max-h-36 overflow-y-auto whitespace-pre-wrap">
                  {selectedJob.payload}
                </div>
              </div>

              {/* Last Run Output */}
              <div className="space-y-1.5">
                <div className="text-[11px] font-mono font-bold text-slate-300 uppercase tracking-wider">
                  Last Execution Output
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 font-mono text-[11px] text-slate-300 max-h-40 overflow-y-auto whitespace-pre-wrap">
                  {selectedJob.lastRunResult || '(No runs executed yet)'}
                </div>
              </div>

              {/* Logs */}
              <div className="space-y-1.5">
                <div className="text-[11px] font-mono font-bold text-slate-300 uppercase tracking-wider">
                  Execution Trace Logs ({selectedJob.logs.length})
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-2.5 font-mono text-[10px] text-slate-400 space-y-1 max-h-36 overflow-y-auto">
                  {selectedJob.logs.map((log, i) => (
                    <div key={i} className="border-b border-slate-850 pb-1">{log}</div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-16 text-slate-500 text-xs font-mono space-y-2">
              <Activity className="w-8 h-8 text-slate-700 mx-auto" />
              <div>Select a scheduled job on the left to inspect detailed payload and execution logs.</div>
            </div>
          )}
        </div>
      </div>

      {/* Register Job Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <form onSubmit={handleRegisterJob} className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg shadow-2xl p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-xs font-bold font-mono text-slate-100 flex items-center gap-2">
                <Plus className="w-4 h-4 text-violet-400" /> Register Temporal / Trigger.dev Schedule
              </h3>
              <button type="button" onClick={() => setShowAddModal(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Job Title / Name:</label>
                <input
                  type="text"
                  value={newJobName}
                  onChange={(e) => setNewJobName(e.target.value)}
                  placeholder="e.g. Sync Code Script or Launch Doc File"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Scheduler Server:</label>
                  <select
                    value={newServer}
                    onChange={(e: any) => setNewServer(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  >
                    <option value="temporal">Temporal.io Server</option>
                    <option value="trigger_dev">Trigger.dev Cloud</option>
                    <option value="embedded">Embedded Scheduler Engine</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Schedule Type:</label>
                  <select
                    value={newScheduleType}
                    onChange={(e: any) => setNewScheduleType(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  >
                    <option value="cron">Cron Expression Syntax</option>
                    <option value="interval">Repeat Interval (Seconds)</option>
                  </select>
                </div>
              </div>

              {newScheduleType === 'cron' ? (
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Cron Expression (e.g. */5 * * * *):</label>
                  <input
                    type="text"
                    value={newCronExpr}
                    onChange={(e) => setNewCronExpr(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-indigo-300 font-mono"
                  />
                </div>
              ) : (
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Interval Seconds:</label>
                  <input
                    type="number"
                    value={newIntervalSec}
                    onChange={(e) => setNewIntervalSec(Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-indigo-300 font-mono"
                  />
                </div>
              )}

              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">
                  Target Code or Local File Path (.py, .exe, .sh to execute; .docx, .pdf to open):
                </label>
                <textarea
                  value={newPayload}
                  onChange={(e) => setNewPayload(e.target.value)}
                  placeholder="Enter raw python/js code OR path like C:\Scripts\tool.exe OR report.docx..."
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono h-28 resize-none"
                />
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowAddModal(false)}
                className="px-3 py-1.5 text-xs font-mono text-slate-400 hover:text-slate-200"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="px-4 py-1.5 bg-violet-600 hover:bg-violet-500 text-white font-mono text-xs font-bold rounded-lg shadow-md transition"
              >
                {isSubmitting ? 'Registering...' : 'Submit Schedule'}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
