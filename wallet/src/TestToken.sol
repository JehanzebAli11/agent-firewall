// SPDX-License-Identifier: MIT
// Test-only ERC20 with EIP-2612 permit. Local anvil use only.
pragma solidity ^0.8.24;

contract TestToken {
    string public name;
    string public symbol;
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public nonces;
    bytes32 public immutable DOMAIN_SEPARATOR;
    bytes32 public constant PERMIT_TYPEHASH =
        keccak256("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)");

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory n, string memory s) {
        name = n;
        symbol = s;
        DOMAIN_SEPARATOR = keccak256(abi.encode(
            keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
            keccak256(bytes(n)), keccak256(bytes("1")), block.chainid, address(this)));
    }

    function mint(address to, uint256 v) external { totalSupply += v; balanceOf[to] += v; emit Transfer(address(0), to, v); }

    function transfer(address to, uint256 v) external returns (bool) { _move(msg.sender, to, v); return true; }

    function approve(address sp, uint256 v) external returns (bool) { allowance[msg.sender][sp] = v; emit Approval(msg.sender, sp, v); return true; }

    function increaseAllowance(address sp, uint256 add) external returns (bool) {
        allowance[msg.sender][sp] += add; emit Approval(msg.sender, sp, allowance[msg.sender][sp]); return true;
    }

    function transferFrom(address f, address to, uint256 v) external returns (bool) {
        uint256 a = allowance[f][msg.sender];
        require(a >= v, "allowance");
        if (a != type(uint256).max) allowance[f][msg.sender] = a - v;
        _move(f, to, v);
        return true;
    }

    function permit(address owner, address spender, uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) external {
        require(block.timestamp <= deadline, "expired");
        bytes32 digest = keccak256(abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR,
            keccak256(abi.encode(PERMIT_TYPEHASH, owner, spender, value, nonces[owner]++, deadline))));
        address rec = ecrecover(digest, v, r, s);
        require(rec != address(0) && rec == owner, "sig");
        allowance[owner][spender] = value;
        emit Approval(owner, spender, value);
    }

    function _move(address f, address to, uint256 v) internal {
        require(balanceOf[f] >= v, "balance");
        balanceOf[f] -= v; balanceOf[to] += v; emit Transfer(f, to, v);
    }
}
