import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Button,
  ConfigProvider,
  Drawer,
  Form,
  Input,
  Modal,
  Progress,
  Radio,
  Select,
  Slider,
  Space,
  Tooltip,
  message
} from 'antd';
import {
  BgColorsOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SearchOutlined,
  SettingOutlined,
  SendOutlined
} from '@ant-design/icons';
import { motion, AnimatePresence } from 'framer-motion';
import '@fontsource/nunito-sans/400.css';
import '@fontsource/nunito-sans/500.css';
import '@fontsource/nunito-sans/600.css';
import '@fontsource/inter/500.css';
import '@fontsource/instrument-serif/400.css';

import { THEME_TOKENS, NAV_ITEMS } from './utils/constants';
import { apiJson, displayText, reportSeedHtml, clampPct, compactNumber, isSensitiveInputError, showSensitiveInputModal, readLastQuery, persistLastQuery } from './utils/helpers';
import { useLoadLatest, useLoadStatus, useObservabilityTrace } from './hooks/useApi';
import usePolling from './hooks/usePolling';
import useSSE from './hooks/useSSE';

import ErrorBoundary from './components/ErrorBoundary';
import CommandDock from './components/CommandDock';
import ConfigDrawer from './components/ConfigDrawer';

import HomeView from './views/HomeView';
import IntelligenceView from './views/IntelligenceView';
import EvidenceView from './views/EvidenceView';
import ReviewView from './views/ReviewView';
import MonitorView from './views/MonitorView';

export default function App() {
  const [active, setActive] = useState('command');
  const [visualTheme, setVisualTheme] = useState(() => window.localStorage.getItem('signal-studio-theme') || 'blue');
  const [coordinatorTask, setCoordinatorTask] = useState(null);
  const [reportTask, setReportTask] = useState(null);
  const [reportEvents, setReportEvents] = useState([]);
  const [reportHtml, setReportHtml] = useState('');
  const [annotations, setAnnotations] = useState([]);
  const [query, setQuery] = useState(() => readLastQuery());
  const queryHydratedFromServer = useRef(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [readoutOpen, setReadoutOpen] = useState(false);
  const [feedbackForm, setFeedbackForm] = useState({ target: 'Overall quality', action: 'Revise', priority: 'Normal', feedback: '' });

  const theme = THEME_TOKENS[visualTheme] || THEME_TOKENS.blue;

  const { latest, setLatest, metadata, feedback, observability, loadingLatest, loadLatest } = useLoadLatest();
  const { system, setSystem, config, setConfig, loadStatus } = useLoadStatus();
  const { observabilityTrace, loading: observabilityLoading, loadObservabilityTrace } = useObservabilityTrace();
  const { startPoll, clearPoll } = usePolling();
  const { startReportStream, clearStream } = useSSE();

  const output = latest || {};
  const synthesis = output.synthesis || {};
  const sourceData = output.source_data || {};
  const queryAgent = sourceData.query_agent || {};
  const insights = synthesis.top_insights || [];
  const risks = synthesis.key_tensions || [];

  const healthScore = useMemo(() => {
    const confidence = clampPct(synthesis.overall_confidence || output.synthesis_confidence || 0);
    const sources = Math.min(100, Number(queryAgent.total_sources || 0));
    const errors = Array.isArray(output.agent_errors) ? output.agent_errors.length : 0;
    return Math.max(0, Math.round(confidence * 0.62 + sources * 0.28 - errors * 12));
  }, [output, queryAgent.total_sources, synthesis.overall_confidence]);

  const toggleVisualTheme = () => {
    setVisualTheme((current) => {
      const next = current === 'blue' ? 'green' : 'blue';
      window.localStorage.setItem('signal-studio-theme', next);
      return next;
    });
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const data = await loadLatest(true);
      if (cancelled) return;
      const serverQuery = displayText(data?.output?.query, '').trim();
      if (!serverQuery) return;
      setQuery((current) => {
        const next = current.trim() || serverQuery;
        persistLastQuery(next);
        return next;
      });
      queryHydratedFromServer.current = true;
    })();
    loadStatus();
    loadObservabilityTrace(true);
    return () => {
      cancelled = true;
      clearPoll();
      clearStream();
    };
  }, []);

  useEffect(() => {
    if (queryHydratedFromServer.current) return;
    const serverQuery = displayText(output.query, '').trim();
    if (!serverQuery) return;
    setQuery((current) => {
      const next = current.trim() || serverQuery;
      persistLastQuery(next);
      return next;
    });
    queryHydratedFromServer.current = true;
  }, [output.query]);

  const handleQueryChange = (event) => {
    const value = event.target.value;
    setQuery(value);
    persistLastQuery(value);
  };

  const runAnalysis = async (extraFeedback = '') => {
    const analysisQuery = query.trim() || displayText(output.query, '');
    if (!analysisQuery) {
      message.warning('Enter an analysis brief first');
      return;
    }
    persistLastQuery(analysisQuery);
    try {
      const data = await apiJson('/api/coordinator/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: analysisQuery, feedback: extraFeedback })
      });
      setCoordinatorTask(data.task);
      setActive('command');
      message.success('Analysis started');
      startPoll(data.task.task_id, async (task) => {
        setCoordinatorTask(task);
        if (['completed', 'error'].includes(task.status)) {
          if (task.status === 'completed') {
            const data = await loadLatest(true);
            const completedQuery = displayText(data?.output?.query, '').trim();
            if (completedQuery) {
              setQuery(completedQuery);
              persistLastQuery(completedQuery);
            }
          }
        }
      });
    } catch (error) {
      if (isSensitiveInputError(error)) {
        showSensitiveInputModal();
        return;
      }
      message.error(error.message || 'Analysis failed to start');
    }
  };

  const generateReport = async () => {
    const started = await startReportStream(query, output, setReportTask, setReportEvents, setReportHtml);
    if (started) setActive('review');
  };

  const saveFeedback = async (runAfter = false) => {
    const text = feedbackForm.feedback.trim();
    if (!text) {
      message.warning('Write a concrete revision request first');
      return;
    }
    try {
      await apiJson('/api/coordinator/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query || output.query || '',
          target: feedbackForm.target,
          action: feedbackForm.action,
          priority: feedbackForm.priority,
          feedback: text,
          thread_id: coordinatorTask?.thread_id || ''
        })
      });
      message.success(runAfter ? 'Feedback saved. Refinement started.' : 'Feedback saved');
      setFeedbackForm((current) => ({ ...current, feedback: '' }));
      setFeedbackOpen(false);
      await loadLatest(true);
      if (runAfter) runAnalysis(text);
    } catch (error) {
      message.error(error.message || 'Feedback could not be saved');
    }
  };

  const startSystem = async () => {
    try {
      const data = await apiJson('/api/system/start', { method: 'POST' });
      message.success(data.message || 'System startup requested');
      loadStatus();
      const latestData = await loadLatest(true);
      const serverQuery = displayText(latestData?.output?.query, '').trim();
      if (serverQuery) {
        setQuery((current) => {
          const next = current.trim() || serverQuery;
          persistLastQuery(next);
          return next;
        });
      }
    } catch (error) {
      message.error(error.message || 'System startup failed');
    }
  };

  const shutdownSystem = () => {
    Modal.confirm({
      title: 'Shut down this workspace?',
      content: 'This stops the running backend services for the current session.',
      okText: 'Shut Down',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          const data = await apiJson('/api/system/shutdown', { method: 'POST' });
          message.success(data.message || 'Shutdown requested');
        } catch (error) {
          message.error(error.message || 'Shutdown request failed');
        }
      }
    });
  };

  return (
    <ConfigProvider theme={{ token: { colorPrimary: theme.primary, borderRadius: 12, fontFamily: 'Inter, sans-serif' } }}>
      <div className={`studio-shell theme-${visualTheme}`}>
        <aside className="studio-nav">
          <div className="brand">
            <div className="brand-mark">S</div>
            <div>
              <strong>Signal Studio</strong>
              <span>Opinion intelligence</span>
            </div>
          </div>
          <nav>
            {NAV_ITEMS.map((item) => (
              <button key={item.key} className={active === item.key ? 'active' : ''} onClick={() => setActive(item.key)}>
                {item.icon}<span>{item.label}</span>
              </button>
            ))}
          </nav>
          <div className="nav-status">
            <span>Quality Score</span>
            <Progress percent={healthScore} strokeColor={theme.primarySoft} trailColor="rgba(255,255,255,.12)" />
            <small>{latest ? 'Latest analysis' : system.started ? 'Ready to run' : 'Start runtime'}</small>
          </div>
        </aside>

        <main className="studio-main">
          <header className="hero-bar compact-hero">
            <div className="hero-lockup">
              <span className="kicker">Signal Studio</span>
              <h1>Sense. Decide.</h1>
            </div>
            <Space wrap className="icon-actions">
              <Tooltip title={`Theme: ${theme.label}`}><Button shape="circle" icon={<BgColorsOutlined />} onClick={toggleVisualTheme} /></Tooltip>
              <Tooltip title="Refresh"><Button shape="circle" icon={<ReloadOutlined />} loading={loadingLatest || observabilityLoading} onClick={() => { loadLatest(false); loadStatus(); loadObservabilityTrace(false); }} /></Tooltip>
              <Tooltip title="Settings"><Button shape="circle" icon={<SettingOutlined />} onClick={() => setConfigOpen(true)} /></Tooltip>
              <Tooltip title="Shutdown"><Button danger shape="circle" icon={<PauseCircleOutlined />} onClick={shutdownSystem} /></Tooltip>
            </Space>
          </header>

          <section className="brief-panel minimal-brief">
            <SearchOutlined />
            <Input value={query} onChange={handleQueryChange} placeholder="Topic" />
            <CommandDock latest={latest} task={coordinatorTask} onRun={() => runAnalysis()} onReport={generateReport} onFeedback={() => setFeedbackOpen(true)} onOpen={() => setActive(latest ? 'intelligence' : 'control')} />
          </section>

          <AnimatePresence mode="wait">
            {active === 'command' && (
              <ErrorBoundary key="command-boundary">
                <HomeView output={output} coordinatorTask={coordinatorTask} theme={theme} risks={risks} queryAgent={queryAgent} synthesis={synthesis} setActive={setActive} />
              </ErrorBoundary>
            )}

            {active === 'intelligence' && (
              <ErrorBoundary key="intelligence-boundary">
                <IntelligenceView output={output} theme={theme} setReadoutOpen={setReadoutOpen} />
              </ErrorBoundary>
            )}

            {active === 'evidence' && (
              <ErrorBoundary key="evidence-boundary">
                <EvidenceView output={output} theme={theme} />
              </ErrorBoundary>
            )}

            {active === 'review' && (
              <ErrorBoundary key="review-boundary">
                <ReviewView output={output} reportTask={reportTask} reportHtml={reportHtml} setReportHtml={setReportHtml} annotations={annotations} setAnnotations={setAnnotations} theme={theme} reportEvents={reportEvents} generateReport={generateReport} />
              </ErrorBoundary>
            )}

            {active === 'control' && (
              <ErrorBoundary key="control-boundary">
                <MonitorView output={output} theme={theme} system={system} coordinatorTask={coordinatorTask} observabilityTrace={observabilityTrace} observability={observability} observabilityLoading={observabilityLoading} loadStatus={loadStatus} loadObservabilityTrace={loadObservabilityTrace} startSystem={startSystem} feedback={feedback} setFeedbackOpen={setFeedbackOpen} metadata={metadata} queryAgent={queryAgent} synthesis={synthesis} />
              </ErrorBoundary>
            )}
          </AnimatePresence>
        </main>
      </div>

      <Drawer open={feedbackOpen} onClose={() => setFeedbackOpen(false)} width={520} title="Revision Request">
        <div className="drawer-stack">
          <Form layout="vertical">
            <Form.Item label="Review target"><Select value={feedbackForm.target} onChange={(value) => setFeedbackForm((current) => ({ ...current, target: value }))} options={['Overall quality', 'Executive readout', 'Evidence grounding', 'Report narrative', 'Risk interpretation'].map((value) => ({ value }))} /></Form.Item>
            <Form.Item label="Requested action"><Radio.Group value={feedbackForm.action} onChange={(event) => setFeedbackForm((current) => ({ ...current, action: event.target.value }))}><Radio.Button value="Review">Review</Radio.Button><Radio.Button value="Revise">Revise</Radio.Button><Radio.Button value="Rerun">Rerun</Radio.Button></Radio.Group></Form.Item>
            <Form.Item label="Priority"><Slider marks={{ 0: 'Normal', 50: 'High', 100: 'Critical' }} step={null} value={{ Normal: 0, High: 50, Critical: 100 }[feedbackForm.priority]} onChange={(value) => setFeedbackForm((current) => ({ ...current, priority: value === 100 ? 'Critical' : value === 50 ? 'High' : 'Normal' }))} /></Form.Item>
            <Form.Item label="Specific request"><Input.TextArea value={feedbackForm.feedback} onChange={(event) => setFeedbackForm((current) => ({ ...current, feedback: event.target.value }))} placeholder="Explain what is wrong, what evidence is missing, or how the report should change" autoSize={{ minRows: 5, maxRows: 8 }} /></Form.Item>
          </Form>
          <Button block size="large" onClick={() => saveFeedback(false)}>Save Request</Button>
          <Button block size="large" type="primary" icon={<SendOutlined />} onClick={() => saveFeedback(true)}>Save and Run Refinement</Button>
        </div>
      </Drawer>

      <Modal open={readoutOpen} onCancel={() => setReadoutOpen(false)} footer={null} width={860} title="Readout Details">
        <div className="readout-modal">
          <p>{displayText(synthesis.summary, 'No synthesis is available yet.')}</p>
          <div className="modal-list">
            {(synthesis.recommended_investigation || []).slice(0, 5).map((item, index) => <div key={index}><span>{(index + 1)}</span>{displayText(item, 'Follow-up')}</div>)}
          </div>
        </div>
      </Modal>

      <ConfigDrawer open={configOpen} onClose={() => setConfigOpen(false)} config={config} setConfig={setConfig} onSaved={loadStatus} />
    </ConfigProvider>
  );
}
