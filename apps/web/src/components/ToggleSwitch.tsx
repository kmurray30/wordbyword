import "./ToggleSwitch.css";

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label: string;
}

// A button-based stand-in for <input type="checkbox"> - see SegmentedControl
// for why native form controls other than the chat textarea are being
// removed from the page.
export function ToggleSwitch({ checked, onChange, disabled, label }: ToggleSwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      className={`toggle-switch${checked ? " toggle-switch--on" : ""}`}
      onClick={() => onChange(!checked)}
    >
      <span className="toggle-switch__track">
        <span className="toggle-switch__thumb" />
      </span>
      {label}
    </button>
  );
}
