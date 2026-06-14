import { Empty } from 'antd';
import { motion } from 'framer-motion';
import { FLOW_STEPS, MICRO_STEPS } from '../utils/constants';
import { displayLog, flowProgressMeta } from '../utils/helpers';

export default function SignalGraph({ output, task, onOpen, theme }) {
  const running = task?.status === 'running';
  const liveDetails = task?.details || {};
  const liveEvidence = Array.isArray(liveDetails.evidence) ? liveDetails.evidence.filter(Boolean).slice(0, 3) : [];
  const completed = running ? flowProgressMeta(task?.progress || 0).normalized : (output?.synthesis ? 100 : 0);
  const meta = running ? flowProgressMeta(completed) : flowProgressMeta(output?.synthesis ? 100 : 0);
  const exactMicroIndex = running
    ? MICRO_STEPS.findIndex((item) => item.stageId === task?.stage && item.name === task?.micro_stage)
    : -1;
  const currentMicroIndex = exactMicroIndex >= 0 ? exactMicroIndex : meta.microIndex;
  const currentStageIndex = exactMicroIndex >= 0 ? MICRO_STEPS[exactMicroIndex].stageIndex : meta.activeIndex;
  const currentStage = FLOW_STEPS[currentStageIndex] || meta.stage;
  const activeIndex = running ? currentStageIndex : output?.synthesis ? FLOW_STEPS.length - 1 : -1;
  const points = [
    { x: 110, y: 230 },
    { x: 285, y: 120 },
    { x: 470, y: 184 },
    { x: 650, y: 84 },
    { x: 820, y: 198 },
    { x: 1010, y: 118 }
  ];
  const d = `M ${points[0].x} ${points[0].y} C 190 112, 205 112, ${points[1].x} ${points[1].y} S 372 270, ${points[2].x} ${points[2].y} S 555 18, ${points[3].x} ${points[3].y} S 724 305, ${points[4].x} ${points[4].y} S 930 66, ${points[5].x} ${points[5].y}`;
  const remaining = Math.max(0, 100 - completed);
  const maskId = 'signal-run-mask';
  const stateClass = running ? 'is-running' : 'is-idle';
  return (
    <div className="signal-stage">
      <div className={`signal-graph ${stateClass} ${output?.synthesis ? 'has-output' : 'is-empty'}`} onClick={onOpen} role="button" tabIndex={0}>
      <svg viewBox="0 0 1120 360" aria-label="Signal flow">
        <defs>
          <linearGradient id="hkuLine" x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%" stopColor={theme.primarySoft} />
            <stop offset="45%" stopColor={theme.primaryMid} />
            <stop offset="100%" stopColor={theme.primaryDark} />
          </linearGradient>
          <filter id="softGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <mask id={maskId} maskUnits="userSpaceOnUse">
            <rect x="0" y="0" width="1120" height="360" fill="black" />
            <path d={d} pathLength="100" stroke="white" strokeWidth="46" strokeLinecap="round" strokeDasharray={`${completed} ${remaining}`} />
          </mask>
        </defs>
        <path className="signal-shadow" d={d} />
        <path className="signal-path" d={d} />
        <path className="signal-idle-pulse" d={d} />
        <path className="signal-run-track" d={d} pathLength="100" strokeDasharray={`${completed} ${remaining}`} />
        <path className="signal-run-pulse" d={d} mask={`url(#${maskId})`} />
        {points.map((point, index) => {
          const active = index <= activeIndex;
          const step = FLOW_STEPS[index];
          return (
            <g className={`signal-node ${active ? 'active' : ''}`} key={step.id} transform={`translate(${point.x} ${point.y})`}>
              <circle r="35" />
              <circle className="node-core" r="12" />
              <text y="62">{step.label}</text>
              <text className="node-sub" y="83">{step.sub}</text>
            </g>
          );
        })}
      </svg>
      </div>
      <div className="graph-status-row">
        <div className="graph-status-main">
          <span>Run State</span>
          <strong>{running ? `${Math.round(completed)}%` : output?.synthesis ? 'Ready' : 'Idle'}</strong>
        </div>
        <div className="graph-trace compact">
          <em><span>Stage</span>{running ? currentStage.label : output?.synthesis ? 'Complete' : 'Standby'}</em>
          <em><span>Step</span>{running ? (task?.micro_stage || MICRO_STEPS[currentMicroIndex]?.name || meta.micro.name) : output?.synthesis ? 'Done' : 'Waiting'}</em>
          <em><span>Progress</span>{running ? `${meta.stagePercent}%` : 'Full path'}</em>
        </div>
      </div>
      <div className="micro-rail" aria-label="Workflow steps">
        {FLOW_STEPS.map((step, stageIndex) => (
          <div className={`micro-group ${stageIndex === currentStageIndex && running ? 'current' : ''}`} key={step.id}>
            <small>{step.label}</small>
            <div className="micro-items">
              {step.micro.map((name, microIndex) => {
                const absoluteIndex = MICRO_STEPS.findIndex((item) => item.stageIndex === stageIndex && item.name === name && item.id.endsWith(`-${microIndex}`));
                const done = running ? absoluteIndex < currentMicroIndex : output?.synthesis;
                const current = running && absoluteIndex === currentMicroIndex;
                return (
                  <b key={`${step.id}-${name}`} className={`${done ? 'done' : ''} ${current ? 'current' : ''}`} title={`${step.label}: ${name}`}>
                    {current && <span>Now</span>}{name}
                  </b>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {running && (
        <div className="live-evidence">
          <strong>{displayLog(liveDetails.message || task?.message || 'Working')}</strong>
          {liveEvidence.map((item, index) => <span key={index}>{displayLog(item)}</span>)}
        </div>
      )}
    </div>
  );
}
