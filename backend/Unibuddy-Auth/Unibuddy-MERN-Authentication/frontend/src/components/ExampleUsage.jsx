import AuthButton from "./AuthButton";

/**
 * Example component showing how to use the AuthButton anywhere in your app
 * 
 * Simply import AuthButton and use it with variant="login" or variant="signup"
 * The modal will automatically open when clicked!
 */

const ExampleUsage = () => {
  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold text-white mb-6">
        How to Use Auth Buttons
      </h2>

      {/* Example 1: Login Button */}
      <div className="mb-8">
        <h3 className="text-lg text-slate-300 mb-3">Login Button:</h3>
        <AuthButton variant="login" />
        <pre className="mt-3 p-4 bg-slate-800 rounded-lg text-sm text-slate-300 overflow-x-auto">
{`import AuthButton from "./components/AuthButton";

<AuthButton variant="login" />`}
        </pre>
      </div>

      {/* Example 2: Signup Button */}
      <div className="mb-8">
        <h3 className="text-lg text-slate-300 mb-3">Signup Button:</h3>
        <AuthButton variant="signup" />
        <pre className="mt-3 p-4 bg-slate-800 rounded-lg text-sm text-slate-300 overflow-x-auto">
{`import AuthButton from "./components/AuthButton";

<AuthButton variant="signup" />`}
        </pre>
      </div>

      {/* Example 3: Custom Styling */}
      <div className="mb-8">
        <h3 className="text-lg text-slate-300 mb-3">With Custom Classes:</h3>
        <AuthButton variant="login" className="w-full" />
        <pre className="mt-3 p-4 bg-slate-800 rounded-lg text-sm text-slate-300 overflow-x-auto">
{`<AuthButton variant="login" className="w-full" />`}
        </pre>
      </div>

      {/* Example 4: Multiple Buttons */}
      <div className="mb-8">
        <h3 className="text-lg text-slate-300 mb-3">Multiple Buttons Together:</h3>
        <div className="flex gap-4">
          <AuthButton variant="login" />
          <AuthButton variant="signup" />
        </div>
        <pre className="mt-3 p-4 bg-slate-800 rounded-lg text-sm text-slate-300 overflow-x-auto">
{`<div className="flex gap-4">
  <AuthButton variant="login" />
  <AuthButton variant="signup" />
</div>`}
        </pre>
      </div>
    </div>
  );
};

export default ExampleUsage;
